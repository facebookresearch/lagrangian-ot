# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn

import copy

from typing import Tuple

from dataclasses import dataclass

from abc import ABC, abstractmethod


plot_cache = {}

@dataclass
class MetricBase(nn.Module):
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


@dataclass
class EuclideanMetric(MetricBase):
    def __call__(self, x):
        assert x.ndim == 1
        D = x.shape[0]
        return jnp.eye(D)


@dataclass
class ScarvelisMetric(MetricBase, ABC):
    div_eps: float = 1e-6
    metric_eps: float = 1e-3

    def __call__(self, x):
        assert x.ndim == 1
        D = x.shape[0]
        v_vals = self.v(x)
        return jnp.eye(D) - (1-self.metric_eps)*jnp.outer(v_vals, v_vals)

    @abstractmethod
    def v(self, x):
        raise NotImplementedError


@dataclass
class CircleMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        norm = jnp.clip(jnp.linalg.norm(x), a_min=self.div_eps)
        return jnp.array([-x[1], x[0]]) / norm

@dataclass
class VeeMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        sign_y = jnp.sign(x[1])
        return jnp.array([1./jnp.sqrt(2), sign_y/jnp.sqrt(2)])

@dataclass
class XMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        a = 1.25 * jax.nn.tanh(jax.nn.relu(x[0]*x[1]))
        b = -1.25 * jax.nn.tanh(jax.nn.relu(-x[0]*x[1]))
        v1 = jnp.array([1./jnp.sqrt(2), 1./jnp.sqrt(2)])
        v2 = jnp.array([1./jnp.sqrt(2), -1./jnp.sqrt(2)])
        return (a*v1 + b*v2) / 1.25

@dataclass
class NeuralNetMetric(MetricBase):
    D = 2

    def setup(self):
        assert self.D == 2
        self.net = nn.Sequential([
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(2)
        ])

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        theta = jnp.arctan2(*self.net(x).squeeze())
        R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                       [jnp.sin(theta), jnp.cos(theta)]])
        Q = jnp.array([[1., 0.],
                       [0., 0.1]])

        A = R.T @ Q @ R
        return A
