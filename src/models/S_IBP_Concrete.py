import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
import pdb
from utils import *

from .common import init_weights, reparametrize, reparametrize_discrete, reparametrize_gaussian

SMALL = 1e-16

class S_IBP_Concrete(nn.Module):
    def __init__(self, max_truncation_level=100, temp=1., alpha0=5., dataset='mnist', hidden=500):
        super(S_IBP_Concrete, self).__init__()
        self.temp = temp
        self.dataset = dataset
        self.truncation = max_truncation_level

        self.fc1_encode = nn.Linear(784, hidden)
        self.fc2_encode = nn.Linear(hidden, self.truncation * 3)

        # generate: deep
        self.fc1_decode = nn.Linear(self.truncation, hidden)
        self.fc2_decode = nn.Linear(hidden, 784)

        a_val = np.log(np.exp(alpha0) - 1) # inverse softplus
        b_val = np.log(np.exp(1.) - 1)
        self.beta_a = nn.Parameter(torch.Tensor(self.truncation).zero_() + a_val)
        self.beta_b = nn.Parameter(torch.Tensor(self.truncation).zero_() + b_val)

        init_weights([self.fc1_encode, self.fc2_encode, self.fc1_decode, self.fc2_decode])

    def encode(self, x):
        return torch.split(self.fc2_encode(
            F.relu(self.fc1_encode(x))
        ), self.truncation, 1)

    def init_bias(self, loader):
        if self.cuda:
            bias = torch.cuda.DoubleTensor(784).zero_()
        else:
            bias = torch.DoubleTensor(784).zero_()
        for batch_idx, (data, _) in enumerate(loader):
            new_sum = data.sum(0)
            if self.cuda:
                new_sum = new_sum.cuda()
            bias += new_sum
        bias = bias / len(loader.dataset)
        self.set_bias(torch.log(bias + SMALL) - torch.log(1. - bias + SMALL))

    def set_bias(self, bias):
        self.fc3_decode.bias.data.copy_(bias)

    def set_temp(self, temp):
        self.temp = temp

    def decode(self, z_discrete):
        return F.sigmoid(self.fc2_decode(
            F.relu(self.fc1_decode(
               z_discrete
            ))
        ))

    def forward(self, x, log_prior=None):
        batch_size = x.size(0)
        truncation = self.beta_a.size(0)
        beta_a = F.softplus(self.beta_a) + 0.01
        beta_b = F.softplus(self.beta_b) + 0.01

        # might be passed in for IWAE
        if log_prior is None:
            log_prior = reparametrize(
                beta_a.view(1, truncation).expand(batch_size, truncation),
                beta_b.view(1, truncation).expand(batch_size, truncation),
                ibp=True, log=True)

        logit_x, mu, logvar = self.encode(x.view(-1, 784))
        logit_post = logit_x + logit(log_prior.exp())

        logsample = reparametrize_discrete(logit_post, self.temp)
        z_discrete = F.sigmoid(logsample) # binary
        z_continuous = reparametrize_gaussian(mu, logvar)

        # zero-temperature rounding
        if not self.training:
            z_discrete = torch.round(z_discrete)

        return self.decode(z_discrete * z_continuous), logsample, logit_post, log_prior, mu, logvar, z_discrete, z_continuous

