#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os
import numpy as np
from pathos import multiprocessing
from typing import List, Tuple, Union, Iterator
import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

__ROOT_DIR = os.path.join( os.path.dirname(__file__), "../")
sys.path.append(__ROOT_DIR)

from distribution.mixture_distribution import MultiVariateGaussianMixture, UniVariateGaussianMixture


class MaskedKLDivLoss(_Loss):

    __EPS = 1E-5

    def forward(self, input_x: torch.Tensor, input_y: torch.Tensor, mask: torch.Tensor):

        if mask.is_floating_point():
            mask = mask.float()
        n_elem = torch.sum(mask, dim=-1)
        batch_loss = torch.sum( mask * input_x * (torch.log(input_x + self.__EPS) - torch.log(input_y + self.__EPS)), dim=-1 ) / n_elem

        if self.reduction == "elementwise_mean":
            loss = torch.mean(batch_loss)
        elif self.reduction == "sum":
            loss = torch.sum(batch_loss)

        return loss


class EmpiricalSlicedWassersteinDistance(_Loss):

    def __init__(self, n_slice, size_average=None, reduce=None, reduction='elementwise_mean'):

        super(EmpiricalSlicedWassersteinDistance, self).__init__(size_average, reduce, reduction)

        self._n_slice = n_slice

    def _sample_circular(self, n_dim, size=None, requires_grad=False):
        if size is None:
            v = np.random.normal(size=n_dim).astype(np.float32)
            v /= np.sqrt(np.sum(v**2))
        else:
            v = np.random.normal(size=n_dim*size).astype(np.float32).reshape((size, n_dim))
            v /= np.linalg.norm(v, axis=1, ord=2).reshape((-1,1))

        t = torch.tensor(data=v, dtype=torch.float, requires_grad=requires_grad)

        return t


    def forward(self, input, target):

        n_mb, n_dim = input.shape

        # sliced wesserstein distance
        loss = torch.tensor(1., dtype=torch.float, requires_grad=True)
        for t in range(self._n_slice):
            t_theta = self._sample_circular(n_dim=n_dim)
            x_t,_ = torch.matmul(input, t_theta).topk(k=n_mb)
            y_t,_ = torch.matmul(target, t_theta).topk(k=n_mb)

            loss = torch.add(loss, torch.mean(torch.pow(x_t-y_t,2)))

        loss = torch.div(loss, self._n_slice)

        return loss


class GMMSlicedWassersteinDistance(object):

    __dtype = np.float32

    def __init__(self, n_dim: int, n_slice: int, n_integral_point: int, inv_cdf_method: str, exponent: int = 2, scale_gradient=True, **kwargs):
        lst_accept = ["analytical","empirical"]
        assert inv_cdf_method in lst_accept, "argument `inv_cdf_method` must be one of these: %s" % "/".join(lst_accept)

        self._n_dim = n_dim
        self._n_slice = n_slice
        self._n_integral = n_integral_point
        self._inv_cdf_method = inv_cdf_method
        self._exponent = exponent
        self._scale_gradient = scale_gradient

        self._integration_point = np.arange(self._n_integral)/self._n_integral + 1. / (2*self._n_integral)
        self._init_extend(**kwargs)

    def _init_extend(self, **kwargs):
        pass

    def _init_grad(self, seq_len):
        g_alpha = np.zeros(seq_len, dtype=self.__dtype)
        g_mu = np.zeros((seq_len, self._n_dim), dtype=self.__dtype)
        g_sigma = np.zeros(seq_len, dtype=self.__dtype)

        return g_alpha, g_mu, g_sigma

    def _sample_circular_distribution(self, size: int):
        v = np.random.normal(size=self._n_dim * size).astype(self.__dtype).reshape((size, self._n_dim))
        v /= np.linalg.norm(v, axis=1, ord=2).reshape((-1,1))
        return v

    def _grad_wasserstein1d(self, p_x: UniVariateGaussianMixture, p_y: UniVariateGaussianMixture) -> (float, np.ndarray, np.ndarray, np.ndarray):
        vec_tau = self._integration_point
        n_integral = vec_tau.size
        inv_cdf_method = self._inv_cdf_method
        exponent = self._exponent

        if inv_cdf_method == "analytical":
            t_x = p_x.inv_cdf(vec_tau)
            t_y = p_y.inv_cdf (vec_tau)
        elif inv_cdf_method == "empirical":
            t_x = p_x.inv_cdf_empirical(vec_tau, n_approx=3*n_integral)
            t_y = p_y.inv_cdf_empirical(vec_tau, n_approx=3*n_integral)
        else:
            raise AssertionError("never happen.")

        # calculate wasserstein distance
        dist = np.mean(np.power(t_x - t_y, exponent))

        # calculate gradients
        vec_grad_alpha = np.zeros_like(p_x._alpha, dtype=self.__dtype)
        vec_grad_mu = np.zeros_like(p_x._mu, dtype=self.__dtype)
        vec_grad_sigma = np.zeros_like(p_x._std, dtype=self.__dtype)
        pdf_x = p_x.pdf(t_x)
        for k in range(p_x._n_k):
            # grad_alpha = \int (F_x_inv - F_y_inv)*F_x_k/P_x
            cdf_x_k = p_x.cdf_component(u=t_x, k=k) / p_x._alpha[k]
            grad_alpha_k = 2*np.mean( (t_x - t_y)*cdf_x_k/pdf_x)

            # grad_mu = \int (F_x_inv - F_y_inv)*P_x_k/P_x
            pdf_x_k = p_x.pdf_component(u=t_x, k=k)
            grad_mu_k = 2*np.mean( (t_x - t_y)*pdf_x_k/pdf_x )

            # grad_sigma = \int (F_x_inv - F_y_inv)*z_k*P_x_k/P_x
            z_k = (t_x - p_x._mu[k])/p_x._std[k]
            grad_sigma_k = 2*np.mean( (t_x - t_y)*z_k*pdf_x_k/pdf_x )

            vec_grad_alpha[k] = grad_alpha_k
            vec_grad_mu[k] = grad_mu_k
            vec_grad_sigma[k] = grad_sigma_k

        # auto scaling
        if self._scale_gradient:
            s_mu = np.mean(np.abs(vec_grad_mu))
            s_alpha = np.mean(np.abs(vec_grad_alpha))
            vec_grad_alpha *= (s_mu / s_alpha)

        return dist, vec_grad_alpha, vec_grad_mu, vec_grad_sigma


    def sliced_wasserstein_distance_single(self, f_x: MultiVariateGaussianMixture, f_y: MultiVariateGaussianMixture, mat_theta: np.ndarray = None) -> (float, np.ndarray, np.ndarray, np.ndarray):

        # initialize
        n_slice = self._n_slice
        dist = 0.
        seq_len = f_x.n_component
        g_alpha, g_mu, g_sigma = self._init_grad(seq_len=seq_len)

        # sample mapping hyperplane
        if mat_theta is None:
            mat_theta = self._sample_circular_distribution(size=n_slice)
        else:
            mat_theta = mat_theta
        # map multi-dimensional distribution into each hyperplane
        lst_p_x = [f_x.radon_transform(vec_theta=theta) for theta in mat_theta]
        lst_p_y = [f_y.radon_transform(vec_theta=theta) for theta in mat_theta]

        # calculate distance in each sliced dimension
        grad_func = lambda p_x, p_y: self._grad_wasserstein1d(p_x=p_x, p_y=p_y)
        iter_grad = map(grad_func, lst_p_x, lst_p_y)

        # take average for disance and gradients
        for theta, (dist_t, g_alpha_t, g_mu_t, g_sigma_t) in zip(mat_theta, iter_grad):
            dist += dist_t
            g_alpha += g_alpha_t
            g_mu += np.expand_dims(g_mu_t, axis=1) * theta
            g_sigma += g_sigma_t

        dist /= n_slice
        g_alpha /= n_slice
        g_mu /= n_slice
        g_sigma /= n_slice

        return dist, g_alpha, g_mu, g_sigma


    def sliced_wasserstein_distance_batch(self,
                                          lst_gmm_x: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                          lst_gmm_y: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]) \
                                            -> (List[float], List[np.ndarray], List[np.ndarray], List[np.ndarray]):

        lst_f_x = [MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma) for alpha, mu, sigma in lst_gmm_x]
        lst_f_y = [MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma) for alpha, mu, sigma in lst_gmm_y]

        mat_theta = self._sample_circular_distribution(size=self._n_slice)

        grad_func = lambda f_x, f_y: self.sliced_wasserstein_distance_single(f_x, f_y, mat_theta)
        iter_grad = map(grad_func, lst_f_x, lst_f_y)

        return tuple(map(list, zip(*iter_grad)))



class GMMSlicedWassersteinDistance_Parallel(GMMSlicedWassersteinDistance):

    __dtype = np.float32
    __num_cpu = multiprocessing.cpu_count()

    def sliced_wasserstein_distance_batch(self,
                                          lst_gmm_x: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                          lst_gmm_y: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]) \
                                            -> (List[float], List[np.ndarray], List[np.ndarray], List[np.ndarray]):

        _pool = multiprocessing.Pool(processes=self.__num_cpu)

        lst_f_x = [MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma) for alpha, mu, sigma in lst_gmm_x]
        lst_f_y = [MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma) for alpha, mu, sigma in lst_gmm_y]
        lst_f = zip(lst_f_x, lst_f_y)

        mat_theta = self._sample_circular_distribution(size=self._n_slice)

        grad_func = lambda f_x_and_y: self.sliced_wasserstein_distance_single(*(f_x_and_y), mat_theta=mat_theta)

        n_batch = len(lst_f_x)
        chunksize = n_batch // self.__num_cpu
        iter_grad = _pool.imap(grad_func, lst_f, chunksize=chunksize)
        obj_ret = tuple(map(list, zip(*iter_grad)))

        _pool.close()

        return obj_ret
