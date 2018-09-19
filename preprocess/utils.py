#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Any

def multiple_sort(key, *values, reverse=False):
    sorter = zip(*sorted(zip(key, *values), reverse=reverse))
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

def len_pad_sort(lst_seq, *lst_values, max_len, padding_value, reverse=True):
    """
    all-in-one function: measure sequence length, pad/trim sequence, sort by the original sequence length

    :param lst_seq: list of the sequence(=list of token)
    :param lst_values: arbitlary number of the list of the accompanying values. e.g. list of the list of PoS-tag
    :param max_len: maximum length of the sequence. shorter sequence will be padded, longer sequence will be trimmed.
    :param padding_value: padding value
    :param reverse: sorting order. DEFAULT:True
    :return: tuple of (list_seq, lst_seq_len, *lst_values)
    """
    lst_seq_len = sequence_length(lst_seq)
    lst_seq = pad_trim_sequence(lst_seq, max_len, padding_value)

    lst_seq_len, lst_seq, *lst_values = multiple_sort(lst_seq_len, lst_seq, *lst_values, reverse=reverse)

    return (lst_seq, lst_seq_len, *lst_values)