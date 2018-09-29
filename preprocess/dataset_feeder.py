#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import math
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Dict, Tuple, Union, Any

from more_itertools import chunked

from .tokenizer import AbstractTokenizer
from .corpora import Dictionary


class AbstractFeeder(object):

    __metaclass__ = ABCMeta

    def __init__(self, n_minibatch=1, validation_split=0.0):
        self._n_mb = n_minibatch
        self._validation_split = validation_split
        self._n_validation = math.ceil(n_minibatch*validation_split)

    @abstractmethod
    def _init_iter_batch(self):
        pass

    def __iter__(self):

        iter_dataset = self._init_iter_batch()
        iter_batch = chunked(iter_dataset, self._n_mb)
        for lst_batch in iter_batch:
            valid = lst_batch[:self._n_validation]
            train = lst_batch[self._n_validation:]

            yield train, valid


class GeneralSequenceFeeder(AbstractFeeder):

    def __init__(self, corpus: Iterable, tokenizer: AbstractTokenizer, dictionary: Dictionary, n_minibatch=1, validation_split=0.0):
        super(__class__, self).__init__(n_minibatch, validation_split)

        self._corpus = corpus
        self._tokenizer = tokenizer
        self._dictionary = dictionary

    def _init_iter_batch(self):

        iter_token = self._tokenizer.tokenize(self._corpus)
        iter_token_idx = self._dictionary.iter_transform(iter_token)

        return iter_token_idx


class SeqToGMMFeeder(AbstractFeeder):

    def __init__(self, corpus: Iterable, tokenizer: AbstractTokenizer, dictionary: Dictionary,
                 dict_lst_gmm_param: Dict[str, Any],
                 convert_var_to_std: bool = True,
                 n_minibatch=1, validation_split=0.0):
        super(__class__, self).__init__(n_minibatch, validation_split)

        self._corpus = corpus
        self._tokenizer = tokenizer
        self._dictionary = dictionary
        self._gmm_param = dict_lst_gmm_param
        self._gmm_param_name = "alpha,mu,scale".split(",")

        if convert_var_to_std:
            self._gmm_param["scale"] = [np.sqrt(v_cov) for v_cov in self._gmm_param["cov"]]
        else:
            self._gmm_param["scale"] = self._gmm_param["cov"]
        del self._gmm_param["cov"]


    def _init_iter_batch(self):

        iter_token = self._tokenizer.tokenize(self._corpus)
        iter_token_idx = self._dictionary.iter_transform(iter_token)
        lst_gmm_param = map(self._gmm_param.get, self._gmm_param_name)

        iter_trainset = zip(iter_token_idx, *lst_gmm_param)

        return iter_trainset
