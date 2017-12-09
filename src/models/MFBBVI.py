from __future__ import print_function
import pdb
import math
import sys
import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *
from bbvi import Beta, BernoulliCV, calculate_cv_from_grads
from .common import init_weights
from training.mf_bbvi import elbo

class MFBBVI(nn.Module):
    """
    This model is an implementation of Chatzis (2014)
    Note that since we use BBVI, we need to the use the REINFORCE method/score function trick in order to get gradients right.
    This version has control variates!
    """
    def __init__(self, dataset='mnist', max_truncation_level=100, alpha0=5., cv=False):
        # Only MNIST / Omniglot supported at the moment
        super(MFBBVI, self).__init__()

        self.truncation = max_truncation_level
        self.alpha0 = alpha0
        self.cv = cv

        self.h_inf = nn.Linear(784, 500)
        self.q_inf = nn.Linear(500, 5 * self.truncation)
        # there's an output for mean, variance, pi_k, a_k, and b_k

        # after the z-normal and z-bernoulli have been multiplied
        self.h_gen = nn.Linear(self.truncation, 500)
        self.x_gen = nn.Linear(500, 784)

        self.stats = [10., 10., 0., 0.,]

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        init_weights([self.h_inf, self.q_inf, self.h_gen, self.x_gen])

    def encode(self, x):
        h1 = self.relu(self.h_inf(x))
        # don't split along batch size, but along the 5*K dimension
        return torch.split(self.q_inf(h1), self.truncation, 1)

    def decode(self, z):
        h3 = self.relu(self.h_gen(z))
        return self.sigmoid(self.x_gen(h3))

    def sample(self, mu, logvar, pi_logit, a, b, cv=False):
        # take samples of the Bernoulli - BBVI
        self.z_probs = self.sigmoid(pi_logit)
        if self.cv and cv:
            self.z_bnp = BernoulliCV(self.a_z_bnp)(self.z_probs)
        else:
            self.z_bnp = BernoulliCV()(self.z_probs)

        # take samples of the normal - reparam
        std = logvar.mul(0.5).exp_()

        cuda = mu.is_cuda
        if cuda:
            newTensor = torch.cuda.DoubleTensor
        else:
            newTensor = torch.DoubleTensor


        eps = Variable(newTensor(std.size()).normal_(), requires_grad=False)
        z_real = eps.mul(std).add_(mu)
       
        self.v = Beta()(a, b)

        if self.cv and self.training:
            self.register_grad_hooks(self.z_probs, a, b)

        return self.z_bnp, z_real, self.v

    def register_grad_hooks(self, z_probs, a, b):
        """
        We want the gradient calculation to be saved for control variates

        This is the f(z) = \nabla_\phi \log q(z) * reward
        So we need to divide by the reward to get just \nabla_\phi \log q(z)
        """
        self.grads = {}
        def save_grad(key, grad):
            self.grads[key] = grad
        z_probs.register_hook(lambda grad: save_grad('z_bnp', grad))

    def feed_loss(self, NLL, KLD_zbnp, KLD_v):
        """
        We just feed in the relevant loss to optimize via REINFORCE
        Note: https://discuss.pytorch.org/t/what-is-action-reinforce-r-doing-actually/1294/13
        explains why we're dividing by batch_size
        """
        batch_size = KLD_zbnp.size()[0]
        NLL = NLL.sum(dim=1).view([batch_size, 1]).repeat(1, self.truncation)
        self.z_bnp.reinforce((KLD_zbnp +  NLL)/ batch_size)
        inv_idx = torch.arange(KLD_zbnp.size(1)-1, -1, -1).long().cuda()
        KL_z_filtered = KLD_zbnp[:, inv_idx].cumsum(1)
        self.v.reinforce((KLD_v + KL_z_filtered) / batch_size)

    def forward(self, x, cv=False, test=False):
        """
        Takes in an x, and returns a sampled x ~ p(x | z), and long with each sample from q
        that makes up the z, and the parameters of the q functions.
        """
        mu, logvar, pi_logit, a, b = self.encode(x.view(-1, 784))

        a = nn.Softplus()(a) + 0.01
        b = nn.Softplus()(b) + 0.01
        
        if cv or test:
            self.a_sum = a.sum().data[0]
            self.b_sum = b.sum().data[0]
        i, j, k, l = a.data.min(), b.data.min(), a.data.max(), b.data.max()
        self.stats = [min(i, self.stats[0]), min(j, self.stats[1]), max(k, self.stats[2]), max(l, self.stats[3])]

        z_bnp, z_real, v = self.sample(mu, logvar, pi_logit, a, b, cv=cv)

        # the other stuff must be returned in order to calculate KL divergence
        return self.decode(z_bnp * z_real), z_bnp, z_real, v, mu, logvar, pi_logit, a, b

    def calculate_control_variates(self, x, log_likelihood, optimizer, num_samples=15):
        """
        Here we calculate the terms f(z) and h(z) - since h(z) = f(z)/ELBO, we calculate it by rescaling.
        """
        repeated = x.repeat(num_samples, 1)
        recon_batch, z_bnp, z_real, v, mu, logvar, pi_logit, a, b = self.forward(repeated, False)
        loss, NLL, KLD_zbnp, KL_v, KL_zreal = elbo(recon_batch, repeated, log_likelihood, z_bnp, z_real, v, mu, logvar, pi_logit, a, b)
        self.feed_loss(-1 * NLL.data, -1. * KLD_zbnp.data, -1. * KL_v.data)
        optimizer.zero_grad()
        loss.sum().backward()

        batch_size = x.size()[0]
        reward = -(KLD_zbnp + NLL.sum(1, True).repeat(1, KLD_zbnp.size()[1]))
        self.a_z_bnp = calculate_cv_from_grads(self.grads['z_bnp'], reward, num_samples, batch_size, self.truncation).detach()
