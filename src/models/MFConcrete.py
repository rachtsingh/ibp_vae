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
from bbvi import Beta

from .common import init_weights, reparametrize_discrete

class MFConcrete(nn.Module):
    """
    This model is an implementation of Chatzis (2014)
    Note that since we use BBVI, we need to the use the REINFORCE method/score function trick in order to get gradients right.
    This version has control variates!
    """
    def __init__(self, dataset='mnist', max_truncation_level=100, temp=1., alpha0=5.):
        super().__init__()

        self.truncation = max_truncation_level
        self.alpha0 = alpha0
        self.temp = temp

        self.h_inf = nn.Linear(784, 500)
        self.q_inf = nn.Linear(500, 5 * self.truncation)
        # there's an output for mean, variance, pi_k, a_k, and b_k

        # after the z-normal and z-bernoulli have been multiplied
        self.h_gen = nn.Linear(self.truncation, 500)
        self.x_gen = nn.Linear(500, 784)

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
    
    def set_temp(self, temp):
        self.temp = temp

    def sample(self, mu, logvar, pi_logit, a, b):
        # get the posterior probability for the Concrete distribution
        logsample = reparametrize_discrete(pi_logit, self.temp)
        z_discrete = F.sigmoid(logsample)
        if not self.training:
            z_discrete = torch.round(z_discrete)

        cuda = mu.is_cuda
        if cuda:
            newTensor = torch.cuda.DoubleTensor
        else:
            newTensor = torch.DoubleTensor

        # take samples of the normal - reparam
        std = logvar.mul(0.5).exp_()
        eps = Variable(newTensor(std.size()).normal_(), requires_grad=False)
        z_real = eps.mul(std).add_(mu)
    
        # we still use BBVI for this - albeit without control variates since it doesn't matter
        self.v = Beta()(a, b)

        return z_discrete, z_real, self.v, logsample

    def feed_loss(self, KLD_v, KLD_zbnp):
        """
        We just feed in the relevant loss to optimize via REINFORCE
        Note: https://discuss.pytorch.org/t/what-is-action-reinforce-r-doing-actually/1294/13
        explains why we're dividing by batch_size
        """
        batch_size = KLD_zbnp.size()[0]
        inv_idx = torch.arange(KLD_zbnp.size(1)-1, -1, -1).long().cuda()
        KL_z_filtered = KLD_zbnp[:, inv_idx].cumsum(1)
        self.v.reinforce(KLD_v / batch_size)

    def forward(self, x):
        """
        Takes in an x, and returns a sampled x ~ p(x | z), and long with each sample from q
        that makes up the z, and the parameters of the q functions.
        """
        mu, logvar, pi_logit, a, b = self.encode(x.view(-1, 784))

        a = nn.Softplus()(a) + 0.01
        b = nn.Softplus()(b) + 0.01

        self.a_sum = a.sum().data[0]
        self.b_sum = b.sum().data[0]

        # note: logsample is just logit(z_discrete), unless we've rounded
        z_discrete, z_real, v, logsample = self.sample(mu, logvar, pi_logit, a, b)

        # the other stuff must be returned in order to calculate KL divergence
        return self.decode(z_discrete * z_real), z_discrete, z_real, v, mu, logvar, pi_logit, a, b, logsample

