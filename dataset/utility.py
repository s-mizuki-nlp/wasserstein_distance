#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

def gen_rotate_2Dmatrix(deg):
    rad = np.pi * deg / 180
    s = np.sin(rad)
    c = np.cos(rad)

    return np.array([c,-s,s,c]).reshape((2, 2))
