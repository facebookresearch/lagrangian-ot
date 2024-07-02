# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

import jax
import jax.numpy as jnp

from dataclasses import dataclass

import stochman

def get_basis(D, num_nodes):
    stochman_spline = stochman.curves.CubicSpline(
        begin=torch.zeros(D), end=torch.ones(D), num_nodes=num_nodes)
    basis = jnp.asarray(stochman_spline.basis.detach())
    return basis


def compute_spline(x, y, basis, params, ts):
    assert x.ndim == 1 and y.ndim == 1
    degree = 4
    D = x.shape[0]
    num_edges = basis.shape[0] // degree
    params = params.reshape(num_edges + 1, D)

    coeffs = (basis @ params).reshape(num_edges, degree, D)
    idx = jnp.floor(ts * num_edges).clip(0, num_edges - 1).astype(jnp.int32)
    power = jnp.arange(0, degree).reshape(1, -1)
    tpow = ts.reshape(-1, 1) ** power
    coeffs_idx = coeffs[idx]
    retval = jnp.expand_dims(tpow, -1) * coeffs_idx
    retval = jnp.sum(retval, axis=-2)
    ts = jnp.expand_dims(ts, -1)
    retval += x*(1-ts) + y*ts
    return retval
