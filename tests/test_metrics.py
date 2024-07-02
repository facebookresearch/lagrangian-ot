# Copyright (c) Meta Platforms, Inc. and affiliates

import jax
import jax.numpy as jnp

import pytest

from lagrangian_ot import manifolds, metrics

def test_circle_metric():
    A_fn = metrics.CircleMetric()

    A = A_fn(jnp.array([-1., 0.]))
    expected_A = jnp.array([[1., 0.], [0., 0.]])
    atol = 1e-2
    assert jnp.allclose(A, expected_A, atol=atol)

    A = A_fn(jnp.array([0., 1.]))
    expected_A = jnp.array([[0., 0.], [0., 1.]])
    assert jnp.allclose(A, expected_A, atol=atol)

def test_vee_metric():
    A_fn = metrics.VeeMetric()

    A = A_fn(jnp.array([0., 0.]))
    expected_A = jnp.array([[0.5, 0.], [0., 1.]])
    atol = 1e-2
    assert jnp.allclose(A, expected_A, atol=atol)

    A = A_fn(jnp.array([1., 1.]))
    expected_A = jnp.array([[0.5, -0.5], [-0.5, 0.5]])
    assert jnp.allclose(A, expected_A, atol=atol)
