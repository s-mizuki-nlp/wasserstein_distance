#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class EmpiricalSlicedWassersteinDistance(_Loss):

    def __init__(self, n_slice, size_average=None, reduce=None, reduction='elementwise_mean'):

        super(EmpiricalSlicedWassersteinDistance, self).__init__(size_average, reduce, reduction)

        self._n_slice = n_slice

    def _sample_circular(self, n_dim, size=None, requires_grad=False):
        if size is None:
            v = np.random.normal(size=n_dim).astype(np.float32)
            v /= np.sqrt(np.sum(v**2))
        else:
            v = np.random.normal(size=n_dim*size).astype(np.float32).reshape((size, n_dim))
            v /= np.linalg.norm(v, axis=1, ord=2).reshape((-1,1))

        t = torch.tensor(data=v, dtype=torch.float, requires_grad=requires_grad)

        return t


    def forward(self, input, target):

        n_mb, n_dim = input.shape

        # slicd wesserstein distance
        loss = torch.tensor(1., dtype=torch.float, requires_grad=True)
        for t in range(self._n_slice):
            t_theta = self._sample_circular(n_dim=n_dim)
            x_t,_ = torch.matmul(input, t_theta).topk(k=n_mb)
            y_t,_ = torch.matmul(target, t_theta).topk(k=n_mb)

            loss = torch.add(loss, torch.mean(torch.pow(x_t-y_t,2)))

        loss = torch.div(loss, self._n_slice)
        # loss = torch.mean(torch.stack(lst_loss))

        return loss