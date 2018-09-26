#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from typing import Union
from collections import Counter
import warnings
import pickle

class Dictionary(object):

    __oov = "__oov__"

    def __init__(self, offset: int=1, oov: bool=True, count: bool=False):

        self._token2id = {}
        self._id2token = {}
        self._offset = offset
        self._count = count
        self._oov = oov
        self._init()

    def _init(self):

        self._token2id = {}
        self._id2token = {}
        self._counter = Counter()

    def fit(self, tokenized_corpus):
        self._init()
        idx = len(self._token2id) + self._offset
        for lst_token in tokenized_corpus:
            for token in lst_token:
                if token not in self._token2id:
                    self._token2id[token] = idx
                    self._id2token[idx] = token
                    idx += 1
            if self._count:
                self._counter.update(lst_token)

        if self._oov:
            self._token2id[self.__oov] = idx
            self._id2token[idx] = self.__oov

    @property
    def n_vocab(self) -> int:
        return len(self._id2token) - self._offset

    @property
    def max_id(self) -> int:
        return max(self._id2token.keys())

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def oov_id(self) -> Union[int, None]:
        if self._oov:
            return self._token2id[self.__oov]
        else:
            return None

    @property
    def vocab(self):
        return self._token2id.keys()

    def save(self, file_path: str):
        if len(self._token2id) == 0:
            warnings.warn("dictionary is empty. did you call `fit()` method?")
        with io.open(file_path, mode="wb") as ofs:
            pickle.dump(self, ofs)

    @classmethod
    def load(cls, file_path: str):
        with io.open(file_path, mode="rb") as ifs:
            obj = pickle.load(ifs)
        obj.__class__ = cls
        return obj

    def token(self, token: str) -> (int, int):
        return self._token_to_id(token), self._counter.get(token, 0)

    def _token_to_id(self, token: str) -> int:
        null = self._token2id[self.__oov] if self._oov else None
        return self._token2id.get(token, null)

    def _id_to_token(self, index: int) -> str:
        return self._id2token.get(index, None)

    def __getitem__(self, item: str) -> int:
        return self._token_to_id(item)

    def transform(self, lst_token):
        return [self._token_to_id(token) for token in lst_token]

    def iter_transform(self, iter_lst_token):
        for lst_token in iter_lst_token:
            yield self.transform(lst_token)

    def inverse_transform(self, lst_index):
        return [self._id_to_token(index) for index in lst_index]

    def iter_inverse_transform(self, iter_lst_index):
        for lst_index in iter_lst_index:
            yield self.inverse_transform(lst_index)