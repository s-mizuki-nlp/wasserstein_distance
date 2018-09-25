#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Any
import numpy as np
import copy

def multiple_sort(key, *values, reverse=False):
    sorter = zip(*sorted(zip(key, *values), key=lambda tup: tup[0], reverse=reverse))
    key_sorted, *values_sorted = (list(t) for t in sorter)
    return (key_sorted, *values_sorted)

def sequence_length(lst_seq):
    return [len(seq) for seq in lst_seq]

def pad_trim_sequence(lst_seq: List[List[Any]], max_len: int, padding_value: int):
    for seq in lst_seq:
        diff = max_len - len(seq)
        if diff > 0:
            seq.extend([padding_value]*diff)
        elif diff < 0:
            seq = seq[:max_len]
        else:
            pass

    return lst_seq

def len_pad_sort(lst_seq, *lst_values, max_len=None, padding_value=0, reverse=True):
    """
    all-in-one function: measure sequence length, pad/trim sequence, sort by the original sequence length

    :param lst_seq: list of the sequence(=list of token)
    :param lst_values: arbitlary number of the list of the accompanying values. e.g. list of the list of PoS-tag
    :param max_len: maximum length of the sequence. shorter/longer sequence will be padded/trimmed, if None, automatically adjusted to the maximum length.
    :param padding_value: padding value. DEFAULT:0
    :param reverse: sorting order. DEFAULT:True
    :return: tuple of (lst_seq_len, list_seq, *lst_values)
    """
    lst_seq = copy.deepcopy(lst_seq)
    lst_seq_len = sequence_length(lst_seq)
    max_len = np.max(lst_seq_len) if max_len is None else max_len
    lst_seq = pad_trim_sequence(lst_seq, max_len, padding_value)
    lst_seq_len, lst_seq, *lst_values = multiple_sort(lst_seq_len, lst_seq, *lst_values, reverse=reverse)

    return (lst_seq_len, lst_seq, *lst_values)