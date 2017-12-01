from __future__ import print_function
import pdb
import math
import sys
import os
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *

# GPU!
from lgamma.beta import Beta
from lgamma.digamma import Digamma

from argparse import Namespace

# the prior is default Beta(alpha_0, 1), but you can change that
def kl_divergence(a, b, prior_alpha = 5., prior_beta = 1., log_beta_prior = np.log(1./5.), num_terms=10, args=Namespace()):
    """
    KL divergence between Kumaraswamy(a, b) and Beta(prior_alpha, prior_beta)
    as in Nalisnick & Smyth (2017) (12)
    - we require you to calculate the log of beta function, since that's a fixed quantity
    """
    digamma = Digamma()
    # digamma = b.log() - 1/(2. * b) - 1./(12 * b.pow(2)) # this doesn't seem to work
    first_term = ((a - prior_alpha)/(a+SMALL)) * (-1 * EULER_GAMMA - digamma(b.view(-1, 1)).view(b.size()) - 1./(b+SMALL))
    second_term = (a+SMALL).log() + (b+SMALL).log() + log_beta_prior
    third_term = -(b - 1)/(b+SMALL)

    if args.cuda:
        sum_term = Variable(torch.cuda.DoubleTensor(a.size()).zero_())
    else:
        sum_term = Variable(torch.DoubleTensor(a.size()).zero_())

    # we should figure out if this is enough
    for i in range(1, num_terms+1):
        beta_ = Beta()
        sum_term += beta_(float(i)/(a.view(-1, 1) + SMALL), b.view(-1, 1)).view(a.size())/(i + a * b)

    return (first_term + second_term + third_term + (prior_beta - 1) * b * sum_term)

def log_density_expconcrete(logalphas, logsample, temp):
    """
    log-density of the ExpConcrete distribution, from 
    Maddison et. al. (2017) (right after equation 26)
    Input logalpha is a logit (alpha is a probability ratio)
    """
    exp_term = logalphas + logsample.mul(-temp)
    log_prob = exp_term + np.log(temp) - 2. * F.softplus(exp_term)
    return log_prob

# here, logsample is an instance of the ExpConcrete distribution, i.e. a y in the paper
def kl_discrete(logit_posterior, logit_prior, logsample, temp, temp_prior):
    """
    KL divergence between the prior and posterior
    inputs are in logit-space
    """
    logprior = log_density_expconcrete(logit_prior, logsample, temp_prior)
    logposterior = log_density_expconcrete(logit_posterior, logsample, temp)
    kl = logposterior - logprior
    return kl

def bce_loss(recon_x, x, args=Namespace()):
    return -(recon_x.log() * x.view(-1, args.D) + (1. - recon_x).log() * (1-x.view(-1,args.D)))

def mse_loss(recon_x, x, args=Namespace()):
    return (recon_x.view(-1, args.D) - x.view(-1, args.D)).pow(2)

def log_sum_exp(arr, dim=0):
    """Apply log-sum-exp to get IWAE loss. It's assumed that the samples vary along the `dim` axis"""
    if arr.__class__ == Variable:
        A = Variable(arr.max(dim)[0].data, requires_grad=False)
    else:
        A = arr.max(dim)[0]
    return A + (arr - A).exp().sum(dim).log()

def print_in_epoch_summary(epoch, batch_idx, batch_size, dataset_size, loss, NLL, KLs):
    kl_string = '\t'.join(['KL({}): {:.3f}'.format(key, val / batch_size) for key, val in KLs.items()])
    print('Train Epoch: {} [{:<5}/{} ({:<2.0f}%)]\tLoss: {:.3f}\tNLL: {:.3f}\t{}'.format(
        epoch, (batch_idx + 1) * batch_size, dataset_size,
        100. * (batch_idx + 1) / (dataset_size / batch_size),
        loss / batch_size,
        NLL / batch_size,
        kl_string))
    sys.stdout.flush()

def print_epoch_summary(epoch, loss):
    print('====> Epoch: {:<3} Average loss: {:.4f}'.format(epoch, loss))
    sys.stdout.flush()
