#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from copy import deepcopy
import numpy as np
from typing import List, Union, Any, Dict, Callable

# l1 distance
def l1_distance(vec_x: np.ndarray, vec_y: np.ndarray):
    return np.sum(np.abs(vec_x - vec_y))

# l2 distance
def l2_distance(vec_x: np.ndarray, vec_y: np.ndarray):
    return np.sum((vec_x - vec_y)**2)

# gaussian kernel function
def gaussian_kernel(mat_x: np.ndarray, distance_function: Callable[[np.ndarray, np.ndarray],float] = l2_distance, gamma: float =0.5):
    n = mat_x.shape[0]
    mat_sim = np.zeros((n,n), dtype=mat_x.dtype)
    for i in range(n):
        mat_sim[i] = [np.exp(-gamma * distance_function(mu, mat_x[i])) for mu in mat_x]

    return mat_sim

def determinantal_probability(mat_sim: np.ndarray, vec_score: np.ndarray) -> float:
    return np.exp(log_determinantal_probability(mat_sim, vec_score))

def log_determinantal_probability(mat_sim: np.ndarray, vec_score: np.ndarray) -> float:
    return np.log(np.linalg.det(mat_sim)) + 2*np.sum(np.log(vec_score))

def top_k_set_greedy_search(mat_sim: np.ndarray, vec_score: np.ndarray, top_k: int) -> List[int]:
    n = mat_sim.shape[0]
    assert mat_sim.shape[0] == mat_sim.shape[1]
    assert mat_sim.shape[0] == vec_score.size
    assert top_k <= n
    if top_k == n:
        return list(range(n))

    lst_selected = []
    while True:
        score = -np.inf
        idx = None
        for i in range(n):
            if i in lst_selected:
                continue
            lst_i = deepcopy(lst_selected)
            lst_i.append(i)
            mat = mat_sim[lst_i,:][:,lst_i]
            vec = vec_score[lst_i]
            score_i = log_determinantal_probability(mat, vec)
            if score_i > score:
                score = score_i
                idx = i
        lst_selected.append(idx)

        if len(lst_selected) == top_k:
            break

    return lst_selected