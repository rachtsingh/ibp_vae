from __future__ import print_function
import pdb
import math
import sys
import os
import numpy as np
from argparse import Namespace

from scipy.special import beta, betaln

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from lgamma.beta import LogBeta
from .common import kl_discrete, log_sum_exp, print_in_epoch_summary

def nll_and_kl(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample, alpha0=5., args=Namespace(), test=False):
    """
    Calculate the NLL and KL divergences as in (27), using the samples (this is differentiable)
    (this is from https://arxiv.org/pdf/1402.3427v4.pdf)
    """
    NLL = -1 * log_likelihood(recon_x, x)
    KLD_zreal = -0.5 * (1. + logvar - mu**2 - logvar.exp())
    
    # note: one is a log probability, the other is a logit!
    log_prior = torch.cumsum((v + SMALL).log(), dim=1)
    logit_posterior = pi_logit
    if not test:
        KLD_zbnp = kl_discrete(logit_posterior, logit(log_prior.exp()), logsample, args.temp, args.temp_prior)
    else:
        pi_posterior = torch.sigmoid(logit_posterior.detach())
        pi_prior = torch.exp(log_prior)
        kl_1 = (pi_posterior + SMALL).log() * z_bnp + (1. - pi_posterior + SMALL).log() * (1. - z_bnp)
        kl_2 = (z_bnp * (pi_prior + SMALL).log()) + ((1 - z_bnp) * (1 - pi_prior).log())
        KLD_zbnp = kl_1 - kl_2

    logbeta = LogBeta()
    a_detached = a.detach()
    b_detached = b.detach()
    KL_v_1 = (a_detached - 1) * (v + SMALL).log() + (b_detached - 1) * (1 - v + SMALL).log() - logbeta(a_detached, b_detached)

    # This is really slow
    # KL_v_1 = (a_detached - 1) * (v + SMALL).log() + (b_detached - 1) * (1 - v + SMALL).log() - Variable(newTensor(betaln(a_detached.data.cpu().numpy(), b_detached.data.cpu().numpy())), requires_grad=False)
    
    KL_v_2 = (alpha0 - 1) * (v + SMALL).log() - betaln(alpha0, 1.)
    KL_v = KL_v_1 - KL_v_2

    return NLL, KLD_zreal, KLD_zbnp, KL_v

def elbo(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample, alpha0=5., args=Namespace(), test=False):
    NLL, KLD_zreal, KLD_zbnp, KL_v = nll_and_kl(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample, alpha0, args, test=test)

    # KLD_zbnp and KL_v for reinforce
    return NLL.sum(1, True) + KLD_zreal.sum(1, True) + KLD_zbnp.sum(1, True) + KL_v.sum(1, True), NLL, KLD_zbnp, KL_v, KLD_zreal

def iwae_loss(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample, alpha0=5., args=Namespace(), num_samples=10):
    NLL, KLD_zreal, KLD_zbnp, KL_v = nll_and_kl(recon_x, x, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample, alpha0, args, True)
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

    for batch_idx, (data, zs) in enumerate(train_loader):
        data = Variable(data.double(), requires_grad=False)
        if args.cuda:
            data = data.cuda()
        data = data.view(-1, 784)

        optimizer.zero_grad()
        recon_batch, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample = model(data)
        loss, NLL, KLD_zbnp, KL_v, KL_zreal = elbo(recon_batch, data, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample, model.alpha0, args, test=False)
        model.feed_loss(-1. * KL_v.data, -1. * KLD_zbnp.data)
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

    return train_loss / len(train_loader.dataset)

def test(test_loader, model, log_likelihood, epoch, mode='validation', args=Namespace(), optimizer=None, num_samples=None):
    model.eval()
    model.double()
    test_loss = 0

    # statistics
    activated = torch.cuda.DoubleTensor(args.truncation, 1)
    a_sum = 0.
    b_sum = 0.

    for batch_idx, (data, zs) in enumerate(test_loader):
        data = Variable(data.double(), requires_grad=False)
        if args.cuda:
            data = data.cuda()
        data = data.view(-1, 784)

        recon_batch, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample = model(data)
        loss, NLL, KLD_zbnp, KL_v, KL_zreal = elbo(recon_batch, data, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample, model.alpha0, args, test=True)
        loss = loss.sum()

        test_loss += loss.data[0]
        activated += z_bnp.data.sum(0, True)
        a_sum += model.a_sum
        b_sum += model.b_sum

    a_mean = a_sum / (len(test_loader.dataset) * model.truncation)
    b_mean = b_sum / (len(test_loader.dataset) * model.truncation)

    if epoch % args.log_epoch == 0:
        print('====> {:<12} Epoch: {} Average loss: {:.4f}\tBeta params: mean: ({:.3f}, {:.3f})'.format(
              mode, epoch, test_loss / len(test_loader.dataset), a_mean, b_mean))

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
        recon_batch, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample = model(data)
        loss = iwae_loss(recon_batch, data, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b, logsample, model.alpha0, args, num_samples)
        loss = loss.data.sum()
        total_iwae_loss += loss

    if epoch % args.log_epoch == 0:
        print('====> Epoch: {} Average {} IWAE loss: {:.4f}'.format(
              epoch, mode, total_iwae_loss / len(test_loader.dataset)))

    return total_iwae_loss / len(test_loader.dataset)
