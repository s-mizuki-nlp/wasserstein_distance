#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import numpy as np
import torch
from torch import nn
from typing import List, Union, NoReturn

def manual_backward(lst_tensor: List[torch.Tensor], lst_gradient: List[Union[torch.Tensor, np.ndarray]]) -> NoReturn:
    assert len(lst_tensor) == len(lst_gradient), "tensor and gradient length must be identical."
    lst_retain_graph = [True]*(len(lst_tensor)-1) + [False]
    for tensor, gradient, retain_graph in zip(lst_tensor, lst_gradient, lst_retain_graph):
        assert tensor.shape == gradient.shape, "dimension size mismatch detected."
        if isinstance(gradient, torch.Tensor):
            v_grad = gradient
        elif isinstance(gradient, np.ndarray):
            v_grad = torch.from_numpy(gradient)
        else:
            raise TypeError("invalid gradient type detected.")

        tensor.backward(gradient=v_grad, retain_graph=retain_graph)
