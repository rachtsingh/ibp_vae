
from __future__ import print_function
import pdb
import math
import sys
import os
import numpy as np
from argparse import Namespace

from scipy.special import betaln

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from lgamma.beta import LogBeta
from .common import log_sum_exp, print_in_epoch_summary

def nll_and_kl(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, alpha0=5.):
    """
    Calculate the NLL and KL divergences as in (27), using the samples (this is differentiable)
    (this is from https://arxiv.org/pdf/1402.3427v4.pdf)
    """
    NLL = -1 * log_likelihood(recon_x, x)
    KLD_zreal = -0.5 * (1. + logvar - mu**2 - logvar.exp())

    pi_prior = torch.cumsum((v.detach() + SMALL).log(), dim=1).exp()
    pi_posterior = torch.sigmoid(pi_logit.detach())

    # the prior here is a function of samples from the posterior of v
    kl_1 = ((pi_posterior + SMALL).log() * z_bnp + (1. - pi_posterior + SMALL).log() * (1. - z_bnp))
    kl_2 = (z_bnp * (pi_prior + SMALL).log() + (1 - z_bnp) * (1 - pi_prior + SMALL).log())
    KLD_zbnp = kl_1 - kl_2

    # logbeta = LogBeta()
    a_detached = a.detach()
    b_detached = b.detach()

    if (a_detached.data < 0).any() or (b_detached.data < 0).any():
        pdb.set_trace()

    # This is really slow
    # way_1 = Variable((a_detached.data.lgamma() + b_detached.data.lgamma() - (a_detached + b_detached).data.lgamma()), requires_grad=False)
    # way_2 = Variable(newTensor(betaln(a_detached.data.cpu().numpy(), b_detached.data.cpu().numpy())), requires_grad=False) 
    # print((way_1.data - way_2.data).abs().max())

    KL_v_1 = (a_detached - 1) * (v + SMALL).log() + (b_detached - 1) * (1 - v + SMALL).log() - Variable((a_detached.data.lgamma() + b_detached.data.lgamma() - (a_detached + b_detached).data.lgamma()), requires_grad=False)
    KL_v_2 = (alpha0 - 1) * (v + SMALL).log() - betaln(alpha0, 1.)
    KL_v = KL_v_1 - KL_v_2

    # perhaps add the regularization to the generation net as a KL divergence here
    return NLL, KLD_zreal, KLD_zbnp, KL_v

def elbo(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, alpha0=5.):
    NLL, KLD_zreal, KLD_zbnp, KL_v = nll_and_kl(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, alpha0=alpha0)

    # KLD_zbnp and KL_v for reinforce
    return NLL.sum(1, True) + KLD_zreal.sum(1, True) + KLD_zbnp.sum(1, True) + KL_v.sum(1, True), NLL, KLD_zbnp, KL_v, KLD_zreal

def iwae_loss(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, alpha0=5., num_samples=10):
    NLL, KLD_zreal, KLD_zbnp, KL_v = nll_and_kl(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, alpha0=alpha0)
    # afterwards, each is [num_samples x batch_size]
    NLL = torch.stack(torch.chunk(NLL, num_samples, 0)).sum(2)
    KLD_zreal = torch.stack(torch.chunk(KLD_zreal, num_samples, 0)).sum(2)
    KLD_zbnp = torch.stack(torch.chunk(KLD_zbnp, num_samples, 0)).sum(2)
    KL_v = torch.stack(torch.chunk(KL_v, num_samples, 0)).sum(2)
    return -log_sum_exp(-(NLL + KLD_zreal + KLD_zbnp + KL_v), 0) + np.log(num_samples)

def train(train_loader, model, log_likelihood, optimizer, epoch, args=Namespace()):
    model.train()
    model.double()
    train_loss = 0

    # statistics
    activated = torch.cuda.DoubleTensor(args.truncation, 1)
    a_sum = 0.
    b_sum = 0.
    model.stats = [10., 10., 0., 0.]

    for batch_idx, (data, zs) in enumerate(train_loader):
        data = Variable(data.double(), requires_grad=False)
        if args.cuda:
            data = data.cuda()
        data = data.view(-1, 784)

        # this lets the model estimate a (from (16))
        if model.cv:
            model.calculate_control_variates(data, log_likelihood, optimizer, num_samples=args.n_cv_samples)
        optimizer.zero_grad()

        recon_batch, z_bnp, z_real, v, mu, logvar, pi_logit, a, b = model(data, cv=model.cv)
        loss, NLL, KLD_zbnp, KL_v, KL_zreal = elbo(recon_batch, data, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, model.alpha0)
        model.feed_loss(-1 * NLL.data, -1. * KLD_zbnp.data, -1. * KL_v.data)
        optimizer.zero_grad()
        loss = loss.sum()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        activated += z_bnp.data.sum(0, True)
        a_sum += model.a_sum
        b_sum += model.b_sum

        if batch_idx % args.log_interval == 0 and epoch % args.log_epoch == 0:
            print_in_epoch_summary(epoch, batch_idx, len(data), len(train_loader.dataset), loss.data[0], NLL.sum().data[0],
                    {'zreal': KL_zreal.sum().data[0], 'beta': KL_v.sum().data[0], 'discrete': KLD_zbnp.sum().data[0]})

    a_mean = a_sum / (len(train_loader.dataset) * model.truncation)
    b_mean = b_sum / (len(train_loader.dataset) * model.truncation)

    if epoch % args.log_epoch == 0:
        print('====> Epoch: {} Average loss: {:.4f}\tBeta params: mean: ({:.3f}, {:.3f})'.format(
              epoch, train_loss / len(train_loader.dataset), a_mean, b_mean))
        print(model.stats)
        sys.stdout.flush()

    return train_loss / len(train_loader.dataset)

def test(test_loader, model, log_likelihood, epoch, mode='validation', args=Namespace(), optimizer=None):
    model.eval()
    model.double()
    test_loss = 0

    # statistics
    activated = torch.cuda.DoubleTensor(args.truncation, 1)
    a_sum = 0.
    b_sum = 0.
    a_var = 0.
    b_var = 0.

    for batch_idx, (data, zs) in enumerate(test_loader):
        data = Variable(data.double(), requires_grad=False)
        if args.cuda:
            data = data.cuda()
        data = data.view(-1, 784)

        # this lets the model estimate a (from (16))
        recon_batch, z_bnp, z_real, v, mu, logvar, pi_logit, a, b = model(data, cv=False, test=True)
        loss, NLL, KLD_zbnp, KL_v, KL_zreal = elbo(recon_batch, data, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, model.alpha0)
        loss = loss.sum()

        test_loss += loss.data[0]
        activated += z_bnp.data.sum(0, True)
        
        a_sum += model.a_sum
        b_sum += model.b_sum
        a_var = a.var().data[0]
        b_var = b.var().data[0]

    a_mean = a_sum / (len(test_loader.dataset) * model.truncation)
    b_mean = b_sum / (len(test_loader.dataset) * model.truncation)

    print('====> {:<12} Average loss: {:.4f}\tBeta params: mean: ({:.3f}, {:.3f}), var: ({:.3f}, {:.3f})'.format(mode, test_loss / len(test_loader.dataset), a_mean, b_mean, a_var, b_var))
    sys.stdout.flush()

    return test_loss / len(test_loader.dataset)

def calculate_iwae_loss(test_loader, model, log_likelihood, epoch, mode='validation', num_samples=10, args=Namespace()):
    model.eval()
    model.double()
    total_iwae_loss = 0.0
    for batch_idx, (data, zs) in enumerate(test_loader):
        data = Variable(data.double(), requires_grad=False)
        if args.cuda:
            data = data.cuda()
        data = data.view(-1, 784)
        data = data.repeat(num_samples, 1)

        # no need for control variates since there's no gradient calculation
        recon_batch, z_bnp, z_real, v, mu, logvar, pi_logit, a, b = model(data)
        loss = iwae_loss(recon_batch, data, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, model.alpha0, num_samples)
        loss = loss.data.sum()
        total_iwae_loss += loss

    if epoch % args.log_epoch == 0:
        print('====> Epoch: {} Average {} IWAE loss: {:.4f}'.format(
              epoch, mode, total_iwae_loss / len(test_loader.dataset)))
        sys.stdout.flush()

    return total_iwae_loss / len(test_loader.dataset)

