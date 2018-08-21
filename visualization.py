#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from scipy.stats import kde
import numpy as np
from matplotlib import pyplot as plt


def kernel_density(mat_X, fig_and_ax=None, vis_range=None, n_mesh_bin=100, overlay_scatter_plot=False, **kwargs):

    n_sample, n_dim = mat_X.shape
    assert n_dim == 2, "visualization isn't available except 2-dimensional distribution."

    # 1. create mesh
    if vis_range is None:
        x_min, y_min = mat_X.min(axis=0)
        x_max, y_max = mat_X.max(axis=0)
    else:
        x_min = y_min = vis_range[0]
        x_max = y_max = vis_range[1]

    mesh_x, mesh_y = np.meshgrid(np.linspace(x_min, x_max, n_mesh_bin), np.linspace(y_min, y_max, n_mesh_bin))
    mesh_xy = np.vstack([mesh_x.flatten(), mesh_y.flatten()]) # shape=(2, n_mesh_bin^2), mesh_xy[:,i] = position(x_i, y_i)

    # 2. calculate z-value as the kernel density
    k = kde.gaussian_kde(mat_X.T)
    value_z = k(mesh_xy) # shape=(n_mesh_bin^2,), value_z[i] = \sum_{n}{k(X_i, X_n)}

    # 3. visualize on canvas
    if fig_and_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_and_ax[0], fig_and_ax[1]

    ax.pcolormesh(mesh_x, mesh_y, value_z.reshape(mesh_x.shape), **kwargs)
    if overlay_scatter_plot:
        ax.scatter(x=mat_X[:,0],y=mat_X[:,1], alpha=0.5, c="black", s=1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    return fig, ax