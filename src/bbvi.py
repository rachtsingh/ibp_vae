"""
TODO: readd tests back here when we figure out the API
"""

import numpy as np
import torch
from torch.autograd.stochastic_function import StochasticFunction
from torch.autograd._functions.stochastic import Bernoulli
from torch.autograd import Variable
from lgamma import internals
import pdb

from utils import *
from scipy.special import beta, digamma

def sample_beta(a, b):
    if torch.cuda.is_available():
        # return internals.beta_sample(a, b)
        return torch.from_numpy(np.random.beta(a.cpu().numpy(), b.cpu().numpy())).cuda()
    else:
        return torch.from_numpy(np.random.beta(a.numpy(), b.numpy()))

def sample_beta_prior(alpha0, size):
    if torch.cuda.is_available():
        return internals.beta_sample(torch.cuda.DoubleTensor(*size).zero_() + alpha0, torch.cuda.DoubleTensor(*size).zero_() + 1)
    else:
        return torch.from_numpy(np.random.beta(alpha0, 1, size=tuple(size))).double()

def slow_beta(a, b):
    return torch.from_numpy(beta(a.numpy(), b.numpy()))

def slow_digamma(x):
    if torch.cuda.is_available():
        return (torch.from_numpy(digamma(x.cpu().numpy()))).cuda()
    else:
        return torch.from_numpy(digamma(x.numpy()))

def fast_digamma(x):
    return internals.polygamma(0, x)

def calculate_cv_from_grads(grads, reward, num_samples, batch_size, truncation):
    h = torch.stack(torch.chunk(grads/reward, num_samples, 0))
    h = h - torch.mean(h, 0, keepdim=True)
    f = torch.stack(torch.chunk(grads, num_samples, 0))
    f = f - torch.mean(f, 0, keepdim=True)
    cov = torch.sum(h * f, 0) / (num_samples - 1.)
    var = (torch.sum(h * h, 0) + SMALL) / (num_samples - 1.)

    return (cov/var)/batch_size

class Beta(StochasticFunction):
    """
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/autograd/_functions/stochastic.py
    but with the Beta distribution
    """
    def __init__(self, control_variates=None):
        super(Beta, self).__init__()
        self.control_variates = control_variates
    
    def forward(self, a, b):
        samples = sample_beta(a, b)
        self.save_for_backward(a, b, samples)
        self.mark_non_differentiable(samples)
        return samples
    
    def backward(self, reward):
        a, b, samples = self.saved_tensors
        grad_a = (samples + SMALL).log() - (fast_digamma(a) - fast_digamma(a + b))
        grad_b = (1. - samples + SMALL).log() - (fast_digamma(b) - fast_digamma(a + b))

        grad_a = -1 * grad_a
        grad_b = -1 * grad_b
        if self.control_variates is not None:
            return grad_a * (reward - self.control_variates[0].data), grad_b * (reward - self.control_variates[1].data)
        else:
            ret = (grad_a * reward, grad_b * reward)
            return grad_a * reward, grad_b * reward

class BernoulliCV(Bernoulli):
    def __init__(self, control_variates=None):
        super(BernoulliCV, self).__init__()
        self.control_variates = control_variates

    def backward(self, reward):
        if self.control_variates is not None:
            return super(BernoulliCV, self).backward(reward - self.control_variates.data)
        else:
            return super(BernoulliCV, self).backward(reward)
