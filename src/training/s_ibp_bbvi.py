"""
The signature for each model is:
`nll_and_kl`: takes in the required parameters, and returns separate expressions for the full (not summed) NLL and KLs
`elbo`: takes in required parameters, runs them through the nll_and_kl, and then returns the scalar loss, and each of the components in a tuple
`iwae_objective`: takes the same + num_samples, returns the scalar loss, and each of the components in a tuple

One note: the scaling is (1./dataset_size), because that term will be counted once per element in the dataset (irrespective of IWAE/ELBO), so it'll
be counted once finally.
"""
from __future__ import print_function
import pdb
import math
import sys
import os
import numpy as np
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from .common import kl_divergence, log_sum_exp, \
        print_in_epoch_summary, print_epoch_summary
from models.common import reparametrize

###
# Methods for our model with BBVI
###

def nll_and_kl(recon_x, x, log_likelihood, a, b, log_prior, logit_post, z_discrete, mu, logvar, dataset_size, args=Namespace()):
    batch_size = x.size()[0]
    NLL = -1 * log_likelihood(recon_x, x)
    KL_zreal = -0.5 * (1. + logvar - mu**2 - logvar.exp())
    KL_beta = kl_divergence(a, b, prior_alpha=args.alpha0, log_beta_prior=np.log(1./args.alpha0), args=args)
    KL_beta = KL_beta.repeat(batch_size, 1) * (1. / dataset_size)
    
    pi_prior = torch.exp(log_prior).detach()
    pi_posterior = torch.sigmoid(logit_post).detach()
    kl_1 = (pi_posterior + SMALL).log() * z_discrete + (1. - pi_posterior + SMALL).log() * (1. - z_discrete)
    kl_2 = (pi_prior + SMALL).log() * z_discrete + (1 - pi_prior + SMALL).log() * (1. - z_discrete)
    KL_discrete = (kl_1 - kl_2)
    return NLL, KL_zreal, KL_beta, KL_discrete

def elbo(recon_x, x, log_likelihood, a, b, log_prior, logit_post, z_discrete, mu, logvar, dataset_size, args=Namespace()):
    NLL, KL_zreal, KL_beta, KL_discrete = nll_and_kl(recon_x, x, log_likelihood, a, b, log_prior, logit_post, z_discrete, mu, logvar, dataset_size, args)
    return (NLL.sum() + KL_zreal.sum() + KL_beta.sum() + KL_discrete.sum()), (NLL, KL_zreal, KL_beta, KL_discrete)

def iwae_loss(recon_x, x, log_likelihood, a, b, log_prior, logit_post, z_discrete, mu, logvar, args, num_samples, dataset_size):
    NLL, KL_zreal, KL_beta, KL_discrete = nll_and_kl(recon_x, x, log_likelihood, a, b, log_prior, logit_post, z_discrete, mu, logvar, dataset_size, args)

    NLL = torch.stack(torch.chunk(NLL, num_samples, 0)).sum(2)
    KL_zreal = torch.stack(torch.chunk(KL_zreal, num_samples, 0)).sum(2)
    KL_discrete = torch.stack(torch.chunk(KL_discrete, num_samples, 0)).sum(2)

    # we throw away the KL_beta until the end
    return (NLL + KL_discrete + KL_zreal).sum(1)

def train(train_loader, model, log_likelihood, optimizer, epoch, args=Namespace()):
    model.train()
    model.double()
    train_loss = 0
    a_sum = 0.
    b_sum = 0.
    for batch_idx, (data, zs) in enumerate(train_loader):
        data = Variable(data.double(), requires_grad=False)
        if args.cuda:
            data = data.cuda()
        if model.cv:
            optimizer.zero_grad()
            model.calculate_control_variates(data, log_likelihood, optimizer, args.n_cv_samples, elbo, len(train_loader.dataset), args)

        recon_batch, log_prior, logit_post, mu, logvar, z_discrete, z_continuous = model(data, cv=model.cv)
        loss, (NLL, KL_zreal, KL_beta, KL_discrete) = elbo(
                recon_batch, data, log_likelihood,
            F.softplus(model.beta_a), F.softplus(model.beta_b),
                log_prior, logit_post, z_discrete, mu, logvar, len(train_loader.dataset), args)
        model.feed_loss(-1 * NLL.data, -1 * KL_discrete.data, -1 * KL_beta.data)
        optimizer.zero_grad()
        loss = loss.sum()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        a_sum += F.softplus(model.beta_a).sum().data[0]
        b_sum += F.softplus(model.beta_b).sum().data[0]

        if batch_idx % args.log_interval == 0 and epoch % args.log_epoch == 0:
            print_in_epoch_summary(epoch, batch_idx, len(data), len(train_loader.dataset), loss.data[0], NLL.sum().data[0], 
                    {'zreal': KL_zreal.sum().data[0], 'beta': KL_beta.sum().data[0], 'discrete': KL_discrete.sum().data[0]})

    if epoch % args.log_epoch == 0:
        print_epoch_summary(epoch, train_loss / len(train_loader.dataset))
        a_mean = a_sum / (len(train_loader.dataset) * model.truncation / args.batch_size)
        b_mean = b_sum / (len(train_loader.dataset) * model.truncation / args.batch_size)
        print('====> Epoch: {} Beta params: mean: ({:.3f}, {:.3f})'.format(epoch, a_mean, b_mean))
        sys.stdout.flush()

    return train_loss / len(train_loader.dataset)

def test(test_loader, model, log_likelihood, epoch, mode='validation', args=Namespace(), optimizer=None):
    model.train()
    model.double()
    train_loss = 0
    a_sum = 0.
    b_sum = 0.
    for batch_idx, (data, zs) in enumerate(test_loader):
        data = Variable(data.double(), requires_grad=False)
        if args.cuda:
            data = data.cuda()
        if model.cv:
            optimizer.zero_grad()
            model.calculate_control_variates(data, log_likelihood, optimizer, args.n_cv_samples, elbo, len(test_loader.dataset), args)

        recon_batch, log_prior, logit_post, mu, logvar, z_discrete, z_continuous = model(data, cv=model.cv)
        loss, (NLL, KL_zreal, KL_beta, KL_discrete) = elbo(
                recon_batch, data, log_likelihood,
            F.softplus(model.beta_a), F.softplus(model.beta_b),
                log_prior, logit_post, z_discrete, mu, logvar, len(test_loader.dataset), args)
        model.feed_loss(-1 * NLL.data, -1 * KL_discrete.data, -1 * KL_beta.data)
        optimizer.zero_grad()
        loss = loss.sum()

        train_loss += loss.data[0]
        a_sum += F.softplus(model.beta_a).sum().data[0]
        b_sum += F.softplus(model.beta_b).sum().data[0]

    if epoch % args.log_epoch == 0:
        print_epoch_summary(epoch, train_loss / len(test_loader.dataset))
        a_mean = a_sum / (len(test_loader.dataset) * model.truncation / args.batch_size)
        b_mean = b_sum / (len(test_loader.dataset) * model.truncation / args.batch_size)
        print('====> {:<12} Epoch: {} Beta params: mean: ({:.3f}, {:.3f})'.format(mode, epoch, a_mean, b_mean))
        sys.stdout.flush()

    return train_loss / len(test_loader.dataset)

def eval_iwae_loss(test_loader, model, log_likelihood, epoch, num_samples=10, args=Namespace(), mode='validation'):
    model.eval()
    model.double()
    
    # initialize the k-set of losses, and sample the stick breaking weights
    losses = torch.zeros(num_samples).double().cuda()
    size = args.batch_size * num_samples
    log_priors = reparametrize(
        F.softplus(model.beta_a).view(1, model.truncation).expand(size, model.truncation),
        F.softplus(model.beta_b).view(1, model.truncation).expand(size, model.truncation),
        ibp=True, log=True)

    for batch_idx, (data, _) in enumerate(test_loader):
        data = Variable(data.double(), requires_grad=False)
        if args.cuda:
            data = data.cuda()
        data = data.repeat(num_samples, 1)
        recon_batch, log_prior, logit_post, mu, logvar, z_discrete, z_continuous = model(data, False, log_priors[:data.size(0)])
        batch_loss = iwae_loss(recon_batch, data, log_likelihood,
            F.softplus(model.beta_a), F.softplus(model.beta_b),
            log_prior, logit_post, z_discrete, mu, logvar, args, num_samples, len(test_loader.dataset))
        losses += batch_loss.data
    
    kl_beta = kl_divergence(F.softplus(model.beta_a), F.softplus(model.beta_b), prior_alpha=args.alpha0, log_beta_prior=np.log(1./args.alpha0), args=args)
    losses += kl_beta.data.sum()

    iwae_loss = (-(log_sum_exp(-losses, dim=0)) + np.log(num_samples))/(len(test_loader.dataset))

    print('====> Importance weighted {:<12} loss (n={}): {:.3f}'.format(mode, num_samples, iwae_loss[0]))
    sys.stdout.flush()
    return iwae_loss[0]
