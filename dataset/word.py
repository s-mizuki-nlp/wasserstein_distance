#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from typing import List, Union

import copy
# from string import ascii_lowercase
import numpy as np
from operator import add, sub
from .utility import gen_rotate_2Dmatrix

class WordTo2DGaussian(object):

    __n_dim = 2
    __n_char = 4

    def __init__(self, lst_mu: Union[List[float],np.ndarray], lst_var: Union[List[float],np.ndarray]):

        self._set_mu = self._init_mu(n_set=self.__n_dim, lst_mu_1d=lst_mu)
        self._op = [add, sub]
        self._cov = self._init_cov(lst_var=lst_var)

        n_mu = len(lst_mu)
        self._n_char_vocab_tup = (n_mu,)*self.__n_dim + (len(self._op), len(self._cov))

    def _init_mu(self, n_set, lst_mu_1d):

        # replicate 1-D mu to n-D
        mat_mu_base = np.tile(lst_mu_1d, (self.__n_dim, 1))
        # create mu_[s=0,1,2,..n_set] by rotating base n-D mu
        delta_deg = 180 / n_set
        arry_mat_mu = np.stack( gen_rotate_2Dmatrix(delta_deg*s).dot(mat_mu_base).T for s in range(n_set) )

        return arry_mat_mu

    def _init_cov(self, lst_var: List[float]):
        return np.stack(np.eye(self.__n_dim)*var for var in lst_var)

    def _random_word_single(self):
        int_seq = [np.random.randint(low=0, high=n_v, size=1) for n_v in self._n_char_vocab_tup]
        return "".join("%d" % c for c in int_seq)

    def random_word(self, size :int=1):
        if size==1:
            return self._random_word_single()
        else:
            return [self._random_word_single() for n in range(size)]

    def word_to_gaussian_mixture(self, word :str):

        lst_char_idx = [int(c) for c in list(word)]
        assert len(lst_char_idx) == self.__n_char, "character length must be %d." % self.__n_char

        alpha = 1.
        mu_1 = self._set_mu[0][lst_char_idx[0]]
        mu_2 = self._set_mu[1][lst_char_idx[1]]
        op = self._op[lst_char_idx[2]]
        cov = self._cov[lst_char_idx[3]]
        mu = op(mu_1, mu_2)

        return alpha, mu, cov

    @property
    def n_dim(self):
        return self.__n_dim

class WordToScalar(object):

    def __init__(self, word_prefix :str, lst_scale :List[float]):
        self._prefix = word_prefix
        self._n_v = len(lst_scale)
        self._scalar = copy.deepcopy(lst_scale)

    def _random_word_single(self):
        return self._prefix + str(np.random.randint(low=0, high=self._n_v))

    def random_word(self, size: int=1) -> Union[str, List[str]]:
        if size==1:
            return self._random_word_single()
        else:
            return [self._random_word_single() for n in range(size)]

    def word_to_scalar(self, word: str) -> float:
        idx = int(word.replace(self._prefix, ""))
        return self._scalar[idx]


class WordToRotationMatrix(object):

    def __init__(self, word_prefix :str, lst_rotation_degree :List[float]):
        self._prefix = word_prefix
        self._n_v = len(lst_rotation_degree)
        self._rot = self._init_rot(lst_rotation_degree=lst_rotation_degree)

    def _init_rot(self, lst_rotation_degree) -> np.ndarray:
        delta_deg = 180 / len(lst_rotation_degree)
        return np.stack(gen_rotate_2Dmatrix(delta_deg*deg) for deg in lst_rotation_degree)

    def _random_word_single(self):
        return self._prefix + str(np.random.randint(low=0, high=self._n_v))

    def random_word(self, size: int=1) -> Union[str, List[str]]:
        if size==1:
            return self._random_word_single()
        else:
            return [self._random_word_single() for n in range(size)]

    def word_to_rotation_matrix(self, word: str) -> np.ndarray:
        idx = int(word.replace(self._prefix, ""))
        return self._rot[idx]