# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import numpy as np

from manifold_ot import manifolds, metrics, data


def test_dist_derivative():
    m = manifolds.MetricManifold(metric=metrics.EuclideanMetric())

    x = jnp.array([-1., 0])
    y = jnp.array([2., 0])

    source_sampler = iter(data.sampler_from_data(jnp.expand_dims(x, axis=0)))
    target_sampler = iter(data.sampler_from_data(jnp.expand_dims(y, axis=0)))
    m.spline_amortizer.train(source_sampler, target_sampler, max_iter=1001)

    d = m.distance(x, y)
    assert jnp.allclose(d, 4.5, atol=0.1)

    grad_dist = jax.grad(m.distance)
    dx = grad_dist(x, y)
    dx_exact = x-y

    assert jnp.allclose(dx, dx_exact, atol=0.1)
