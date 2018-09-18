#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from abc import ABCMeta, abstractmethod

class AbstractLoader(object):

    __metaclass__ = ABCMeta

    def __init__(self, file_path, n_minibatch=1):
        self._path = file_path
        self._n_mb = n_minibatch

    @abstractmethod
    def __iter__(self):
        pass



class TextLoader(AbstractLoader):

    def __init__(self, file_path, strip=None):
        super(TextLoader, self).__init__(file_path, n_minibatch=0)
        self._strip = strip

    def __iter__(self):
        with io.open(self._path, mode="r") as ifs:
            for line in ifs:
                yield line.strip(self._strip)


class MinibatchTextLoader(AbstractLoader):

    def __init__(self, file_path, n_minibatch, strip=None):
        super(MinibatchTextLoader, self).__init__(file_path, n_minibatch)
        self._strip = strip

    def __iter__(self):
        with io.open(self._path, mode="r") as ifs:
            lst_ret = []
            for line in ifs:
                lst_ret.append(line.strip(self._strip))
                if len(lst_ret) >= self._n_mb:
                    yield lst_ret
                    lst_ret = []
            if len(lst_ret) > 0:
                yield lst_ret