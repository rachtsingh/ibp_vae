import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
from bbvi import BernoulliCV, calculate_cv_from_grads
import pdb
from utils import *

from .common import init_weights, reparametrize, reparametrize_gaussian

class S_IBP_BBVI(nn.Module):
    """
    This is the same model as above (essentially, 2-layer MLP with gaussian + binary hidden variables), but with BBVI used for the binary latent variables
    """
    def __init__(self, max_truncation_level=100, alpha0=5., dataset='mnist', hidden=500, cv=False):
        super(S_IBP_BBVI, self).__init__()
        self.dataset = dataset
        self.truncation = max_truncation_level
        self.cv = cv 

        self.fc1_encode = nn.Linear(784, hidden)
        self.fc2_encode = nn.Linear(hidden, self.truncation * 3)

        # generate: deep
        self.fc1_decode = nn.Linear(self.truncation, hidden)
        self.fc2_decode = nn.Linear(hidden, 784)

        a_val = np.log(np.exp(alpha0) - 1) - 2. # inverse softplus
        b_val = np.log(np.exp(1.) - 1) - 2.
        self.beta_a = nn.Parameter(torch.Tensor(self.truncation).zero_() + a_val)
        self.beta_b = nn.Parameter(torch.Tensor(self.truncation).zero_() + b_val)

        init_weights([self.fc1_encode, self.fc2_encode, self.fc1_decode, self.fc2_decode])

    def encode(self, x):
        """
        Returns the mean, variance, and binary logits
        """
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
        self.set_bias(torch.log(bias + 1e-8) - torch.log(1. - bias + 1e-8))

    def set_bias(self, bias):
        self.fc3_decode.bias.data.copy_(bias)

    def decode(self, z_discrete):
        return F.sigmoid(self.fc2_decode(
            F.relu(self.fc1_decode(
               z_discrete
            ))
        ))

    def register_grad_hooks(self, z_probs):
        self.grads = {}
        def save_grad(key, grad):
            self.grads[key] = grad
        z_probs.register_hook(lambda grad: save_grad('z_bnp', grad))

    def forward(self, x, cv=False, log_prior=None):
        batch_size = x.size(0)
        truncation = self.beta_a.size(0)
        beta_a = F.softplus(self.beta_a)
        beta_b = F.softplus(self.beta_b)

        if log_prior is None:
            log_prior = reparametrize(
                beta_a.view(1, truncation).expand(batch_size, truncation),
                beta_b.view(1, truncation).expand(batch_size, truncation),
                ibp=True, log=True)
        logit_x, mu, logvar = self.encode(x.view(-1, 784))

        # now a logit
        logit_post = logit_x + logit(log_prior.exp())
        self.z_probs = F.sigmoid(logit_post)

        # BBVI 
        self.register_grad_hooks(self.z_probs)
        if cv:
            self.z_bnp = BernoulliCV(self.a_z_bnp)(self.z_probs)
        else:
            self.z_bnp = BernoulliCV()(self.z_probs)

        z_continuous = reparametrize_gaussian(mu, logvar)
        
        return self.decode(self.z_bnp * z_continuous), log_prior, logit_post, mu, logvar, self.z_bnp, z_continuous
    
    def feed_loss(self, NLL, KLD_zbnp, KL_beta):
        """
        We just feed in the relevant loss to optimize via REINFORCE
        Note: https://discuss.pytorch.org/t/what-is-action-reinforce-r-doing-actually/1294/13
        explains why we're dividing by batch_size
        """
        batch_size = KLD_zbnp.size()[0]
        # make them all the right size
        NLL = NLL.sum(dim = 1).view([batch_size, 1]).repeat(1, self.truncation)
        self.z_bnp.reinforce((KLD_zbnp + NLL) / batch_size)

    def calculate_control_variates(self, x, log_likelihood, optimizer, num_samples, elbo, dataset_size, args):
        """
        We calculate gradients, and then estimate a_z_bnp (the control variate)
        """
        repeated = x.repeat(num_samples, 1)
        recon_batch, log_prior, logit_post, mu, logvar, z_discrete, z_continuous = self.forward(repeated, cv=False)
        loss, (NLL, KL_zreal, KL_beta, KL_discrete) = elbo(
                recon_batch, repeated, log_likelihood,
                F.softplus(self.beta_a), F.softplus(self.beta_b),
                log_prior, logit_post, z_discrete, mu, logvar, dataset_size, args)

        self.feed_loss(-1 * NLL.data, -1. * KL_discrete.data, -1 * KL_beta.data)
        optimizer.zero_grad()
        loss.sum().backward()

        batch_size = x.size()[0]

        # cumsum the Beta KL since the probs are calculated like that
        reward = -(KL_discrete + NLL.sum(1, True).repeat(1, KL_discrete.size()[1]))
        self.a_z_bnp = calculate_cv_from_grads(self.grads['z_bnp'], reward, num_samples, batch_size, self.truncation).detach()

