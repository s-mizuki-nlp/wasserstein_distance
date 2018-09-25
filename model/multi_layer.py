#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from torch import nn


class MultiDenseLayer(nn.Module):

    def __init__(self, n_dim_in, n_dim_out, n_dim_hidden, n_hidden, activation_function):
        """
        multi-layer dense neural network with artibrary activation function
        output = Dense(iter(Activation(Dense())))(input)

        :param n_dim_in: input dimension size
        :param n_dim_out: output dimension size
        :param n_dim_hidden: hidden layer dimension size
        :param n_hidden: number of layers
        :param activation_function: activation function. e.g. torch.relu
        """
        super(MultiDenseLayer, self).__init__()

        self._n_hidden = n_hidden
        self._lst_dense = []
        for k in range(n_hidden):
            n_in = n_dim_in if k==0 else n_dim_hidden
            n_out = n_dim_out if k==(n_hidden-1) else n_dim_hidden
            self._lst_dense.append(nn.Linear(n_in, n_out))
        self._activation = activation_function
        self._layers = nn.ModuleList(self._lst_dense)

    def forward(self, x):

        for k, dense in enumerate(self._layers):
            if k == 0:
                h = self._activation(dense(x))
            elif k == (self._n_hidden-1):
                h = dense(h)
            else:
                h = self._activation(dense(h))

        return h