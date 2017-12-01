from __future__ import print_function
import argparse
import torch
import torch.cuda
from torch.autograd import Variable
from scipy.special import polygamma, beta
import math
import sys
import numpy as np

from functions.beta import Beta
from functions.digamma import Digamma

import unittest
from torch.autograd import gradcheck

tensorType = torch.cuda.DoubleTensor

class TestDigammaGrads(unittest.TestCase):
    def test_many_times(self):
        input = (Variable(tensorType(50,50).uniform_(), requires_grad=True),)
        self.assertTrue(gradcheck(Digamma(), input, eps=1e-6, atol=1e-3))

# class TestBetaGrads(unittest.TestCase):
#     def test_many_times(self):
#         a = Variable(tensorType(100, 100).uniform_() * 10, requires_grad=True)
#         b = Variable(tensorType(100, 100).uniform_() * 10, requires_grad=True)
#         result = gradcheck(Beta(), (a, b), eps=1e-6, atol=1e-3)
#         self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
