#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal, norm
from scipy.misc import logsumexp
from scipy import optimize
from matplotlib import pyplot as plt

vector = np.array
matrix = np.ndarray
tensor = np.ndarray

class MultiVariateGaussianMixture(object):

    __EPS = 1E-5

    def __init__(self, vec_alpha: vector, mat_mu: matrix, tensor_cov: tensor = None, vec_std: vector = None):
        self._n_k = len(vec_alpha)
        self._n_dim = mat_mu.shape[1]
        self._alpha = vec_alpha - self.__EPS
        self._ln_alpha = np.log(self._alpha)
        self._mu = mat_mu
        if tensor_cov is not None:
            self._cov = tensor_cov
        elif vec_std is not None:
            self._cov = np.stack([(std**2) * np.eye(self._n_dim) for std in vec_std])
        else:
            raise AttributeError("either `tensor_cov` or `vec_std` must be specified.")
        self._validate()

    def _validate(self):
        msg = "number of mixture component mismatch."
        assert len(self._alpha) == self._n_k, msg
        assert self._mu.shape[0] == self._n_k, msg
        assert self._cov.shape[0] == self._n_k, msg

        msg = "dimensionality mismatch."
        assert self._mu.shape[1] == self._n_dim, msg
        assert self._cov.shape[1] == self._n_dim, msg
        assert self._cov.shape[2] == self._n_dim, msg

    @property
    def n_component(self):
        return self._n_k

    @property
    def n_dim(self):
        return self._n_dim

    @classmethod
    def random_generation(cls, n_k: int, n_dim: int, covariance_type="spherical", mu_range=None, cov_range=None):
        lst_available_covariance_type = "identity,spherical,diagonal".split(",")
        msg = "argument `covariance_type must be one of those: %s" % "/".join(lst_available_covariance_type)
        assert covariance_type in lst_available_covariance_type, msg

        rng_mu = [-2, 2] if mu_range is None else mu_range
        rng_cov = [0.2, 0.5] if cov_range is None else cov_range

        vec_alpha = np.random.dirichlet(alpha=np.ones(n_k)*n_k)
        mat_mu = np.random.uniform(low=rng_mu[0], high=rng_mu[1], size=n_k*n_dim).reshape((n_k, n_dim))
        if covariance_type == "identity":
            tensor_cov = np.array([np.eye(n_dim) for k in range(n_k)])
        elif covariance_type == "spherical":
            tensor_cov = np.array([np.random.uniform(low=rng_cov[0], high=rng_cov[1])*np.eye(n_dim) for k in range(n_k)])
        elif covariance_type == "diagonal":
            tensor_cov = np.array([np.diag(np.random.uniform(low=rng_cov[0], high=rng_cov[1], size=n_dim)) for k in range(n_k)])
        else:
            raise NotImplementedError("unexpected input.")

        return cls(vec_alpha, mat_mu, tensor_cov)

    @classmethod
    def to_tensor(cls, lst_of_tuple, normalize_alpha=False):
        alpha = np.array([tup[0] for tup in lst_of_tuple])
        if normalize_alpha:
            alpha /= np.sum(alpha)
        mu = np.stack(tup[1] for tup in lst_of_tuple)
        cov = np.stack(tup[2] for tup in lst_of_tuple)

        return alpha, mu, cov

    @classmethod
    def to_tuple(cls, vec_alpha, mat_mu, tensor_cov):
        lst_ret = []
        n_k = len(vec_alpha)
        for k in range(n_k):
            lst_ret.append((vec_alpha[k], mat_mu[k], tensor_cov[k]))

        return lst_ret

    @property
    def n_component(self):
        return self._n_k

    def density_plot(self, fig_and_ax=None, vis_range=None, n_mesh_bin=100, **kwargs):
        assert self._n_dim == 2, "visualization isn't available except 2-dimensional distribution."

        rng_default = np.max(np.abs(self._mu)) + 2. * np.sqrt(np.max(self._cov))
        rng = [-rng_default, rng_default] if vis_range is None else vis_range

        mesh_x, mesh_y = np.meshgrid(np.linspace(rng[0], rng[1], n_mesh_bin), np.linspace(rng[0], rng[1], n_mesh_bin))
        mesh_xy = np.vstack([mesh_x.flatten(), mesh_y.flatten()])

        if fig_and_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_and_ax[0], fig_and_ax[1]

        value_z = self.pdf(mesh_xy.T)

        ax.pcolormesh(mesh_x, mesh_y, value_z.reshape(mesh_x.shape), **kwargs)

        return fig, ax

    def pdf(self, vec_x: vector):
        prob = 0.
        for k in range(self._n_k):
            prob += self._alpha[k]*multivariate_normal.pdf(vec_x, self._mu[k], self._cov[k])
        return prob

    def logpdf(self, vec_x: vector):
        if vec_x.ndim == 1:
            v_ln_prob = self._ln_alpha + np.array([multivariate_normal.logpdf(vec_x, self._mu[k], self._cov[k]) for k in range(self._n_k)])
            ln_prob = logsumexp(v_ln_prob)
        elif vec_x.ndim == 2:
            n_size = vec_x.shape[0]
            m_ln_prob = np.empty(shape=(n_size, self._n_k), dtype=np.float)
            for k in range(self._n_k):
                m_ln_prob[:,k] = self._ln_alpha[k] + multivariate_normal.logpdf(vec_x, self._mu[k], self._cov[k])
            ln_prob = logsumexp(m_ln_prob, axis=1)
        return ln_prob

    def random(self, size: int, shuffle=True):
        """
        generate random samples from the distribution.

        :param size: number of samples
        :param shuffle: randomly permute generated samples(DEFAULT:True)
        :return: generated samples
        """
        vec_r_k = np.random.multinomial(n=size, pvals=self._alpha)
        mat_r_x = np.vstack([np.random.multivariate_normal(mean=self._mu[k], cov=self._cov[k], size=size_k) for k, size_k in enumerate(vec_r_k)])
        if shuffle:
            np.random.shuffle(mat_r_x)
        return mat_r_x

    def radon_transform(self, vec_theta: vector):
        mu_t = self._mu.dot(vec_theta) # mu[k]^T.theta
        std_t = np.sqrt(vec_theta.dot(self._cov).dot(vec_theta)) # theta^T.cov[k].theta
        return UniVariateGaussianMixture(self._alpha, mu_t, std_t)

class UniVariateGaussianMixture(object):

    __eps = 1E-5

    def __init__(self, vec_alpha: vector, vec_mu: vector, vec_std: vector):
        self._n_k = len(vec_alpha)
        self._n_dim = 1
        self._alpha = vec_alpha.copy() - self.__eps
        self._ln_alpha = np.log(self._alpha)
        self._mu = vec_mu.copy()
        self._std = vec_std.copy()
        self._cov = np.square(self._std)
        self._emp_x = np.empty(0)
        self._emp_q = np.empty(0)
        self._validate()

    def _validate(self):
        msg = "number of mixture component mismatch."
        assert self._alpha.size == self._n_k, msg
        assert self._mu.size == self._n_k, msg
        assert self._std.size == self._n_k, msg

        msg = "invalid parameter range."
        assert all(self._std > 0.), msg

    def _normalize(self, u: float):
        return (u - self._mu)/self._std

    def _gen_empirical_sequence(self, n_sample):
        self._emp_x = np.sort(self.random(size=n_sample))
        # accurate
        self._emp_q = self.cdf(self._emp_x)
        # fast
        # self._emp_q = np.arange(n_sample)/n_sample + 1. / (2*n_sample)

    @property
    def n_component(self):
        return self._n_k

    def cdf(self, u: vector):
        if isinstance(u, float):
            prob = np.sum(self._alpha * norm.cdf(self._normalize(u)))
        else:
            prob = np.zeros_like(u)
            for k in range(self._n_k):
                vec_z = (u - self._mu[k]) / self._std[k]
                prob += self._alpha[k] * norm.cdf(vec_z)
        return prob

    def pdf(self, u: vector):
        if isinstance(u, float) or isinstance(u, int):
            # component-wise vectorization
            prob = np.sum(self._alpha * norm.pdf(self._normalize(u))/self._std)
        elif isinstance(u, np.ndarray):
            # sample-wise vectorization
            prob = np.zeros_like(u)
            for k in range(self._n_k):
                vec_z = (u - self._mu[k]) / self._std[k]
                prob += self._alpha[k] * norm.pdf(vec_z) / self._std[k]
        return prob

    def pdf_component(self, u: vector, k: int):
        alpha_k = self._alpha[k]
        mu_k = self._mu[k]
        std_k = self._std[k]
        return alpha_k * norm.pdf(u, mu_k, std_k)

    def cdf_component(self, u: vector, k: int):
        alpha_k = self._alpha[k]
        mu_k = self._mu[k]
        std_k = self._std[k]
        return alpha_k * norm.cdf(u, mu_k, std_k)

    def logpdf(self, u: float):
        # component-wise vectorization
        v_ln_prob = self._ln_alpha + norm.logpdf(self._normalize(u)) - np.log(self._std)
        ln_prob = logsumexp(v_ln_prob)
        return ln_prob

    def logcdf(self, u: float):
        v_ln_prob = self._ln_alpha + norm.logcdf(self._normalize(u))
        ln_prob = logsumexp(v_ln_prob)
        return ln_prob

    def grad_pdf(self, u):
        # \nabla \sum_k{\alpha_k * CDF(z_{u,k})}
        grad = np.sum( self._alpha/self._std * norm.pdf(self._normalize(u)) )
        return grad

    def _inv_cdf(self, q: float):

        x0_t = np.sum(self._alpha*self._mu)
        std_t_mean = np.sum(self._alpha*self._std)
        ln_q = np.log(q)

        def func(u):
            # log(\sum_k{\alpha_k * CDF(z_{u,k})})
            ret = self.logcdf(u) - ln_q
            # ret = self.cdf(u) - q
            return ret

        # newton method: it is unstable
        # root = optimize.newton(func=func, fprime=grad_func, x0=x0_t)
        # brent method
        a, b = x0_t-10.*std_t_mean, x0_t+10.*std_t_mean
        root = optimize.brenth(f=func, a=a, b=b, xtol=1e-5)
        return root

    def inv_cdf(self, q: vector):
        if isinstance(q, float) or isinstance(q, int):
            return self._inv_cdf(q)
        elif isinstance(q, np.ndarray):
            return np.vectorize(self._inv_cdf)(q)
        else:
            raise NotImplementedError("unsupported input type.")

    def inv_cdf_empirical(self, q: vector, n_approx=1000):
        if self._emp_x.size != n_approx:
            self._gen_empirical_sequence(n_sample=n_approx)
        # return the interpolated empirical quantile values
        return np.interp(q, xp=self._emp_q, fp=self._emp_x)

    def random(self, size: int, shuffle=True):
        """
        generate random samples from the distribution.

        :param size: number of samples
        :param shuffle: randomly permute generated samples(DEFAULT:True)
        :return: generated samples
        """
        vec_r_k = np.random.multinomial(n=size, pvals=self._alpha)
        vec_r_x = np.concatenate([np.random.normal(loc=self._mu[k], scale=self._std[k], size=size_k) for k, size_k in enumerate(vec_r_k)])
        if shuffle:
            np.random.shuffle(vec_r_x)
        return vec_r_x
