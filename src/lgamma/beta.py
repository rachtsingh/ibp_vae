import torch
import torch.cuda
from .internals import polygamma, lgamma
from scipy.special import polygamma as polygamma_
from scipy.special import digamma as digamma_
from scipy.special import gammaln
from scipy.special import beta as beta_
import numpy as np

class Beta(torch.autograd.Function):
    def forward(self, a, b):
        # beta_ab = (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
        beta_ab = (a.lgamma() + b.lgamma() - (a + b).lgamma()).exp()
        self.save_for_backward(a, b, beta_ab)
        return beta_ab

    def backward(self, grad_output):
        a, b, beta_ab = self.saved_tensors
        digamma_ab = polygamma(0, a + b)
        return grad_output * beta_ab * (polygamma(0, a) - digamma_ab), grad_output * beta_ab * (polygamma(0, b) - digamma_ab)

class LogBeta(torch.autograd.Function):
    def forward(self, a, b):
        logbeta_ab = a.lgamma() + b.lgamma() - (a + b).lgamma() 
        self.save_for_backward(a, b)
        return logbeta_ab

    def backward(self, grad_output):
        a, b = self.saved_tensors
        digamma_ab = polygamma(0, a + b)
        return grad_output * (polygamma(0, a) - digamma_ab), grad_output * (polygamma(0, b) - digamma_ab)
