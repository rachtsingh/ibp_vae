"""
A pure PyTorch-implementation of the mean-field VI algorithm of Doshi-Velez et. al.
http://ai.stanford.edu/~tadayuki/papers/doshivelez-miller-vangael-teh-tech-report09.pdf

This code is still a WIP - there seems to be a lot of numerical instability around the choice of sigma_n.
"""

import torch
import numpy as np
import numpy.random as npr
from scipy.special import digamma as dg
# from lgamma import internals
# we won't import autograd, since we're not using it - everything is a Tensor

LOG_2PI = 1.8378770664093453


def generate_gg_blocks():
    A = np.zeros([4, 36])
    A[0, [1, 6, 7, 8, 13]] = 1
    A[1, [3, 4, 5, 9, 11, 15, 16, 17]] = 1
    A[2, [18, 24, 25, 30, 31, 32]] = 1
    A[3, [21, 22, 23, 28, 34]] = 1
    return A


def generate_random_A(feature_count, dim_count, sigma_a=10):
    A = sigma_a * npr.normal(0, 1, [feature_count, dim_count])
    return A


def generate_data(data_count, data_type, sigma=0.1, sigma_a=5):
    # data_type can be 'gg' for the basic blocks, 'finite-random' for
    # a random A but a finite Z (same size as gg), and
    # 'infinite-random' for a simulated Z from the IBP and a random A
    dim_count = 36
    if data_type == 'gg':
        A = generate_gg_blocks()
        feature_count = A.shape[0]
        dim_count = A.shape[1]
        Z = npr.binomial(1, .25, [data_count, feature_count])
    if data_type == 'finite-random':
        feature_count = 4
        A = generate_random_A(feature_count, dim_count)
        Z = npr.binomial(1, .25, [data_count, feature_count])
    if data_type == 'infinite-random':
        from simulate import sample_Z_restaurant
        Z, feature_count_by_data = sample_Z_restaurant(data_count, 2)
        feature_count = Z.shape[1]
        A = generate_random_A(feature_count, dim_count)

    # add noise to the data and return
    data_set = np.dot(Z, A) + sigma * npr.normal(0, 1, [data_count, dim_count])
    return data_set, Z, A


def digamma(x):
    # return internals.polygamma(0, x)
    return torch.Tensor(dg(x.numpy()))


def compute_E_logstick(tau):
    # tau: [2 x K] stick breaking parameters. This an implementation of (9) from the paper
    # but it calculates it for all k simultaneously, and returns a K-vector with the values
    K = tau.size()[1]
    dgt_1 = digamma(tau[0])
    dgt_2 = digamma(tau[1])
    dgt_sum = digamma(tau.sum(0))
    prefix = torch.cat((torch.zeros(1), dgt_1.cumsum(0)), 0)[:-1]
    full_scores = torch.exp(
        digamma(tau[1]) + prefix - dgt_sum.cumsum(0)).view(1, K).repeat(K, 1).tril()
    # checked that this is the right division, but recheck
    q = full_scores / full_scores.sum(1).view(K, 1)

    # now we calculate (9)
    first_term = (q * dgt_2.view(K, 1)).sum(1)

    # we construct a 'sum past and including' matrix, then shift *left* by 1
    inverse_idx = torch.arange(K - 1, -1, -1).long()
    reverse_cumsum = torch.t(torch.t(q)[inverse_idx].cumsum(0)[inverse_idx])
    second_term = (torch.cat([torch.zeros(K, 1), reverse_cumsum], 1)[
                   :, 1:] * dgt_1.view(K, 1)).sum(1)

    third_term = (reverse_cumsum * dgt_sum.view(K, 1)).sum(1)

    temp_q = q.clone()
    temp_q[q == 0] += 1.
    fourth_term = (temp_q * temp_q.log()).sum(1)

    return first_term + second_term - third_term - fourth_term, q


def compute_elbo(data_set, alpha, sigma_a, sigma_n, phi, Phi, nu, tau):
    """
    Compute the ELBO for the VI model. The variables are from the paper.

    data_set: [N x D]
    alpha: float
    sigma_a: float
    sigma_n: float
    phi: [K x D] # means
    Phi: [K x D] # covariances - but it is isotropic, so just store the individual variances
    nu: [N x K] # bernoulli mean
    tau: [2 x K] # stick breaking Beta parameters
    """
    N, D = data_set.size()
    K = len(phi)

    # these are individual terms of equation (8):
    term_1 = (np.log(alpha) + (alpha - 1) * (digamma(tau[0]) -
                                             digamma(tau.sum(0)))).sum()

    # we have to use the multinomial approximation for this
    hard_term, _ = compute_E_logstick(tau)
    digamma_cumsum = (digamma(tau[1]) -
                      digamma(tau.sum(0))).cumsum(0)  # K-dim
    term_2 = (nu * digamma_cumsum.view(1, K) +
              (1 - nu) * hard_term.view(1, K)).sum()

    # note: first term is scaled by K since it's a scalar, second term is sum
    term_3 = (-0.5 * K * D * (LOG_2PI + 2 * np.log(sigma_a)) -
              (0.5 / sigma_a**2) * (Phi.sum() + phi.pow(2).sum()))

    # this is inside the right term, from left to right
    internal_term = (data_set.pow(2).sum() - 2 * (nu * (data_set @ torch.t(phi))).sum() +
                     2 * (torch.bmm(nu.unsqueeze(2), nu.unsqueeze(1)).sum() * (phi @ torch.t(phi))).tril(-1).sum() +
                     (nu.sum(0) * (Phi.sum(1) + phi.pow(2).sum(1))).sum())

    term_4 = (-0.5 * K * N * (LOG_2PI + 2 * np.log(sigma_n)) -
              (0.5 / sigma_a**2) * (internal_term))

    term_5 = (((tau[0].lgamma() + tau[1].lgamma()) - tau.sum(0).lgamma()).sum() -
              ((tau[0] - 1) * digamma(tau[0]) + (tau[1] - 1) * digamma(tau[1])).sum() +
              ((tau.sum(0) - 2) * digamma(tau.sum(0))).sum())

    # this might be really expensive, I'm unsure
    term_6 = (0.5 * (D * K * (LOG_2PI + 1) + Phi.log().sum()) -
              (nu * nu.log() + (1 - nu) * (1 - nu).log()).sum())

    return term_1 + term_2 + term_3 + term_4 + term_5 + term_6


def vi(data_set, alpha, sigma_a, sigma_n, phi, Phi, nu, tau):
    """
    Compute one mean-field update step as in section 5.2.
    As a reminder:

    data_set: [N x D]
    alpha: float
    sigma_a: float
    sigma_n: float
    phi: [K x D] # means
    Phi: [K x D] # covariances - but it is isotropic, so just store the individual variances
    nu: [N x K] # bernoulli mean
    tau: [2 x K] # stick breaking Beta parameters
    """
    N, D = data_set.size()
    K = len(phi)

    # update Phi
    Phi = (1. / (1. / sigma_a**2. + nu.sum(0) / sigma_n**2.)
           ).view(K, 1) * torch.ones(K, D)

    # update phi
    phi = (1/sigma_n**2.) * (torch.t(nu) @ (data_set - (nu @ phi)) + (phi * nu.sum(0).view(K, 1))) * Phi

    # update nu
    # each of the first 3 terms is K-dimensional
    E_logstick, q = compute_E_logstick(tau)
    nu_frak = (((digamma(tau[1]) - digamma(tau.sum(0))).cumsum(0) - E_logstick -
                0.5 / (sigma_n**2.) * (Phi.sum(1) + phi.pow(2).sum(1))).view(1, K) +
               1. / (sigma_a**2.) * (torch.t(phi @ torch.t(data_set)) -
               (nu @ phi @ torch.t(phi) + nu * (phi.pow(3).sum(1).view(1, K)))))
    nu = torch.sigmoid(nu_frak)

    # update tau
    matrix = nu.sum(0).view(K, 1) * q
    nu_cumsum = N - nu.sum(0).cumsum(0)
    for k in range(K):
        tau[0][k] = alpha + nu_cumsum[k] + matrix[k:, k:].sum()
        tau[1][k] = 1 + (nu_cumsum * q[:, k]).sum()

    return data_set, alpha, sigma_a, sigma_n, phi, Phi, nu, tau


def main():
    N = 100
    D = 36
    K = 10
    data_set, _, _ = generate_data(N, 'gg')
    data_set = torch.from_numpy(data_set).float()
    alpha = 5.
    sigma_a = 5
    sigma_n = 0.5
    phi = torch.zeros(K, D).uniform_()
    Phi = torch.zeros(K, D).uniform_()
    nu = torch.sigmoid(torch.randn(N, K)) / 10.
    tau = 1 + torch.randn(2, K).pow(2)
    for i in range(10):
        data_set, alpha, sigma_a, sigma_n, phi, Phi, nu, tau = vi(
            data_set, alpha, sigma_a, sigma_n, phi, Phi, nu, tau)
        print(compute_elbo(data_set, alpha, sigma_a, sigma_n, phi, Phi, nu, tau))


main()
