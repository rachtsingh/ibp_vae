"""
Common functions for all models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
import pdb
from utils import *

SMALL = 1e-16

def init_weights(weights):
    for layer in weights:
        torch.nn.init.normal(layer.weight.data, 0, 0.001)
        if layer.bias is not None:
            layer.bias.data.zero_()

# note that we're only sampling len(a), but it gets made to len(a) + 1 in reparamatrize
def kumaraswamy_sample(a, b):
    u = a.data.clone().uniform_(0.001, 0.999)
    u = Variable(u, requires_grad=False)
    # return (1. - u.pow(1./b)).pow(1./a)
    return (1. - u.log().div(b+SMALL).exp() + SMALL).log().div(a+SMALL).exp()

# NOTE: if the input is 24-dimensional, then the output is 25-dimensional
def reparametrize(a, b, ibp=False, log=False):
    v = kumaraswamy_sample(a, b)
    batch_size = a.size()[0]
    cuda = v.is_cuda
    if cuda:
        newTensor = torch.cuda.DoubleTensor
    else:
        newTensor = torch.DoubleTensor

    if ibp:
        # IBP: no need to sum to 1
        v_term = (v+SMALL).log()
        logpis = torch.cumsum(v_term, dim=1)
    else:
        # offset the vs
        v_term = torch.cat([(v+SMALL).log(), Variable(newTensor(batch_size).view(-1, 1).zero_(), requires_grad=False)], 1)

        # offset the 1 - vs
        inv_term = torch.cumsum(torch.cat([Variable(newTensor(batch_size).view(-1, 1).zero_(), requires_grad=False), (1. - v + SMALL).log()], 1), dim=1)
        logpis = v_term + inv_term

    if log:
        return logpis
    else:
        return logpis.exp()

# returns samples from a ExpConcrete(alpha, temp) distribution
def reparametrize_discrete(logalphas, temp):
    """
    input:  logit, output: logit
    """
    uniform = Variable(logalphas.data.clone().uniform_(1e-4, 1. - 1e-4),  requires_grad = False)
    logistic = torch.log(uniform) - torch.log(1. - uniform)
    logsample = (logalphas + logistic) / temp
    return logsample

def reparametrize_gaussian(mu, logvar):
    noise = Variable(mu.data.clone().normal_(0, 1), requires_grad=False)
    return mu + (noise * logvar.exp())

