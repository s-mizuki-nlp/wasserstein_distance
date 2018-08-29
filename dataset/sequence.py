#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import numpy as np
import pandas as pd
import itertools
from . import grammar_template

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

            # DEBUG
            print("".join(lst_seq))

            # check if every non-terminal symbol has been expanded
            is_all_terminal = all([s in self.__terminal for s in lst_seq])

            if is_all_terminal:
                break

        return lst_seq