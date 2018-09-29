#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Any, Union
import numpy as np
import copy
import torch

def multiple_sort(key, *values, reverse=False):
    sorter = zip(*sorted(zip(key, *values), key=lambda tup: tup[0], reverse=reverse))
    key_sorted, *values_sorted = (list(t) for t in sorter)
    return (key_sorted, *values_sorted)

def sequence_length(lst_seq):
    return [len(seq) for seq in lst_seq]

def array_length(lst_array: List[np.ndarray], dim: int = 0):
    return [array.shape[dim] for array in lst_array]

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

def pad_numpy_sequence(lst_array: List[np.ndarray], max_len: int = None, dim: int = 0, pad_value: float =0.0) -> np.ndarray:
    """
    pad and stack the list of variable-size numpy array

    :param lst_array: list of variable-size numpy array
    :param max_len: maximum size of the dimension. shorter dimension will be padded with `pad_value`. if None, automatically adjusted to the maximum size.
    :param dim: padding dimension. DEFAULT:0
    :param pad_value: padding value. DEFAULT:0.0
    :return: padded and stacked numpy array with shape[dim] = max_len
    """
    n = len(lst_array)
    n_dim = lst_array[0].ndim
    lst_arr_len = array_length(lst_array, dim)
    max_len = np.max(lst_arr_len) if max_len is None else max_len

    # construct padding operator
    lst_pad_op = [[(0,0)]*n_dim for _ in range(n)]
    for pad_op, arr_len in zip(lst_pad_op, lst_arr_len):
        pad_op[dim] = (0, max_len - arr_len)

    # pad and stack
    ret = np.stack([np.pad(array, pad_op, mode="constant", constant_values=pad_value) for array, pad_op in zip(lst_array, lst_pad_op)])

    return ret

def pack_padded_sequence(packed_array: Union[np.ndarray, torch.Tensor], lst_seq_len: Union[List[int], np.ndarray], dim: int = 0) -> List[np.ndarray]:
    """
    pack the fixed-size padded numpy array or pytorch tensor into the list of variable-size numpy array

    :param packed_array: fixed-size numpy array to be packed
    :param lst_seq_len: list of the dimension size of each array
    :param dim: packing dimension. DEFAULT:0(=2nd dimension in fixed-size array)
    :return: list of the variable-size packed numpy array
    """
    if isinstance(packed_array, torch.Tensor):
        _packed_array = packed_array.data.numpy()
    else:
        _packed_array = packed_array

    return [array.take(indices=range(seq_len), axis=dim) for array, seq_len in zip(_packed_array, lst_seq_len)]