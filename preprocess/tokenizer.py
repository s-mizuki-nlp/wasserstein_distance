#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from abc import ABCMeta, abstractmethod
from typing import List

class AbstractTokenizer(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def tokenize_single(self, sentence: str) -> List[str]:
        pass

    def tokenize(self, corpus):
        if iter(corpus) is iter(corpus):
            raise TypeError("`corpus` must be a container.")
        for sentence in corpus:
            yield self.tokenize_single(sentence)


class CharacterTokenizer(AbstractTokenizer):

    def __init__(self, separator: str=" "):
        self._sep = separator

    def tokenize_single(self, sentence):
        return sentence.split(self._sep)