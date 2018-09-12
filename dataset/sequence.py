#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Union
import os, sys, io
import numpy as np
import pandas as pd
import itertools
from . import grammar_template
from .word import WordTo2DGaussian, WordToScalar, WordToRotationMatrix

class PCFG(object):

    __grammer = grammar_template.simple_pcfg
    __bos = grammar_template.bos
    __terminal = grammar_template.terminal
    __col_left = "s_0"
    __col_right = ["s_1","s_2"]
    __col_prob = "prob"

    def __init__(self):
        self._grammer, symbols = self._init_grammer()
        self._non_terminal = symbols - self.__terminal

    def _init_grammer(self):

        with io.StringIO(self.__grammer) as ifs:
            df = pd.read_table(ifs, header=0)
            df.fillna(value="", inplace=True)

        symbols = set(filter(bool, df.select_dtypes(exclude=float).values.ravel()))

        return df, symbols

    def _expand_symbol(self, index):
        ret = list(filter(bool, self._grammer.loc[index, self.__col_right].values.ravel()))
        return ret

    def _random_select(self, symbol):
        cand_idx = self._grammer[self.__col_left] == symbol
        if np.sum(cand_idx) > 1:
            s_prob = self._grammer.loc[cand_idx, self.__col_prob]
            cand_idx = np.random.choice(s_prob.index, p=s_prob.values, size=1)

        return self._expand_symbol(index=cand_idx)

    def random_sequence(self):

        lst_seq = [self.__bos]
        while True:
            for i, s in enumerate(lst_seq):
                if s in self._non_terminal:
                    lst_symbol_new = self._random_select(symbol=s)
                    lst_seq[i] = lst_symbol_new

            # flatten again
            lst_seq = list(itertools.chain.from_iterable(lst_seq))

            # check if every non-terminal symbol has been expanded
            is_all_terminal = all([s in self.__terminal for s in lst_seq])

            if is_all_terminal:
                break

        return lst_seq

    @property
    def terminal_symbol(self):
        return self.__terminal


class SentenceToGMM(PCFG):

    __end_of_component = grammar_template.word

    def __init__(self, lst_mu, lst_var, lst_scale, lst_rotation_degree, dtype=np.float32):
        super(SentenceToGMM, self).__init__()
        self._to_word = {
            "w":WordTo2DGaussian(lst_mu=lst_mu, lst_var=lst_var),
            "s":WordToScalar(word_prefix="s", lst_scale=lst_scale),
            "r":WordToRotationMatrix(word_prefix="r", lst_rotation_degree=lst_rotation_degree)
        }
        self._eoc = self.__end_of_component
        self._n_dim = self._to_word["w"].n_dim
        self._dtype = dtype

        self._validate()

    def _validate(self):
        msg = "there is undefined symbol: %s"
        for symbol in self._to_word.keys():
            assert symbol in super(SentenceToGMM, self).terminal_symbol, msg % symbol

        assert self._n_dim == 2, "currently it only supports 2-dimensional GMM."

    def _count_mixture_component(self, lst_symbol):
        return np.sum(s in self._eoc for s in lst_symbol)

    def _symbol_to_word(self, lst_symbol: List[str]) -> List[str]:
        lst_word = []
        for s in lst_symbol:
            word = self._to_word[s].random_word()
            lst_word.append(word)
        return lst_word

    def random_sentence(self, component_min=0, component_max=np.inf):

        while True:
            lst_symbol = super(SentenceToGMM, self).random_sequence()
            n_word = self._count_mixture_component(lst_symbol)
            if n_word < component_min:
                continue
            if n_word > component_max:
                continue
            break

        lst_word = self._symbol_to_word(lst_symbol)

        return lst_symbol, lst_word

    def _init_component(self):
        scale = 1.0
        rotation = np.eye(self._n_dim, dtype=self._dtype)
        return scale, rotation

    def sentence_to_gaussian_mixture(self, lst_symbol, lst_word):

        n_c = self._count_mixture_component(lst_symbol)
        vec_alpha = np.zeros(n_c, dtype=self._dtype)
        mat_mu = np.zeros(shape=(n_c, self._n_dim), dtype=self._dtype)
        tensor_cov = np.zeros(shape=(n_c, self._n_dim, self._n_dim), dtype=self._dtype)

        # initialize
        idx = 0
        scale, rotation = self._init_component()
        for symbol, word in zip(lst_symbol, lst_word):
            word_class = self._to_word[symbol]

            if symbol == "w":
                # set up component
                alpha, mu, cov = word_class.word_to_gaussian_mixture(word)
                alpha = alpha*scale
                mu = rotation.dot(mu)
                vec_alpha[idx] = alpha
                mat_mu[idx] = mu
                tensor_cov[idx] = cov

                # reset status
                scale, rotation = self._init_component()
                idx += 1
            elif symbol == "s":
                scale = scale * word_class.word_to_scalar(word)
            elif symbol == "r":
                rot_new = word_class.word_to_rotation_matrix(word)
                rotation = rotation.dot(rot_new)
            else:
                raise NotImplementedError()

        # normalize alpha
        vec_alpha /= np.sum(vec_alpha)

        return vec_alpha, mat_mu, tensor_cov