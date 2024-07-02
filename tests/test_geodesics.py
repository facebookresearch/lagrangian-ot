# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import numpy as np

from lagrangian_ot import geodesics, manifolds, metrics, data


def test_euclidean_geodesic():
    m = manifolds.MetricManifold(metric=metrics.EuclideanMetric())
    x = jnp.array([-1., 0])
    y = jnp.array([0., 1.])

    source_sampler = iter(data.sampler_from_data(jnp.expand_dims(x, axis=0)))
    target_sampler = iter(data.sampler_from_data(jnp.expand_dims(y, axis=0)))
    m.spline_amortizer.train(source_sampler, target_sampler, max_iter=1001)

    geodesic = m.path(x, y)
    atol = 0.3
    true_geodesic = jnp.linspace(x, y, len(geodesic))
    assert jnp.allclose(geodesic, true_geodesic, atol=atol)


def test_geodesic_solve():
    m = manifolds.MetricManifold(metric=metrics.CircleMetric())

    x = jnp.array([-1., 0])
    y = jnp.array([1, 0.])

    source_sampler = iter(data.sampler_from_data(jnp.expand_dims(x, axis=0)))
    target_sampler = iter(data.sampler_from_data(jnp.expand_dims(y, axis=0)))
    m.spline_amortizer.train(source_sampler, target_sampler, max_iter=1001)

    atol = 1e-3
    geodesic = m.path(x, y)
    assert jnp.allclose(geodesic[0], x, atol=atol)
    assert jnp.allclose(geodesic[-1], y, atol=atol)

    # the geodesic should be on a circle
    norms = jnp.linalg.norm(geodesic, axis=1)
    atol = 0.1
    assert jnp.allclose(norms, 1., atol=atol)
