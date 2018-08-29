#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io

import os, sys, io
# from string import ascii_lowercase
import numpy as np
from operator import add, sub
from .utility import gen_rotate_2Dmatrix

class WordTo2DGaussian(object):

    __n_dim = 2
    __n_char = 4

    def __init__(self):

        self._n_char_vocab = {
            "mu":5,
            "op":2,
            "cov":5
        }
        self._set_mu = self._init_mu(n_set=self.__n_dim, n_diff=self._n_char_vocab["mu"], mu_rng=[-1., 1.])
        self._op = [add, sub]
        self._cov = self._init_cov(n_diff=self._n_char_vocab["cov"], scale_rng=[0.1, 0.5])

        self._n_char_vocab_tup = (self._n_char_vocab["mu"],)*self.__n_dim + (self._n_char_vocab["op"], self._n_char_vocab["cov"])

    def _init_mu(self, n_set, n_diff, mu_rng):

        vec_mu_base = np.linspace(*mu_rng, n_diff)
        #  mat_mu_base[i] = lst_mat_mu_[s=0][i]
        mat_mu_base = np.vstack([vec_mu_base]*self.__n_dim).T
        # create mu_[s=0,1,2,..n_set]
        delta_deg = 180 / n_set
        arry_mat_mu = np.stack( gen_rotate_2Dmatrix(delta_deg*s).dot(mat_mu_base.T).T for s in range(n_set) )

        return arry_mat_mu

    def _init_cov(self, n_diff, scale_rng):
        vec_scale = np.linspace(*scale_rng, n_diff)
        return np.stack(np.eye(self.__n_dim)*s for s in vec_scale)

    def _random_word_single(self):
        int_seq = [np.random.randint(low=0, high=n_v, size=1) for n_v in self._n_char_vocab_tup]
        return "".join("%d" % c for c in int_seq)

    def random_word(self, size :int=1):
        if size==1:
            return self._random_word_single()
        else:
            return [self._random_word_single() for n in range(size)]

    def transform_to_gaussian(self, word :str):

        lst_char_idx = [int(c) for c in list(word)]
        assert len(lst_char_idx) == self.__n_char, "character length must be %d." % self.__n_char

        alpha = 1.
        mu_1 = self._set_mu[0][lst_char_idx[0]]
        mu_2 = self._set_mu[1][lst_char_idx[1]]
        op = self._op[lst_char_idx[2]]
        cov = self._cov[lst_char_idx[3]]
        mu = op(mu_1, mu_2)

        return alpha, mu, cov
