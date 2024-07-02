# Copyright (c) Meta Platforms, Inc. and affiliates

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import copy

import jax
import jax.numpy as jnp
from jax import nn
import numpy as np

import flax


@dataclass
class LagrangianPotentialBase(flax.linen.Module):
    D: int = 2

    M_bounds = (0., 0.01)
    temp_bounds = (1e-1, 1e-2)

    def setup(self):
        self.M = self.param('M', lambda key: jnp.full((1,), self.M_bounds[1]))
        self.temp = self.param('temp', lambda key: jnp.full((1,), self.temp_bounds[1]))

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @classmethod
    def get_annealed_params(cls, t):
        assert 0 <= t and t <= 1
        if 1.-t < 1e-3:
            t = 1.
        elif t < 1e-3:
            t = 0.
        else:
            t = nn.sigmoid(10.*(t-0.5))

        M_start, M_end = cls.M_bounds
        temp_start, temp_end = cls.temp_bounds
        new_M = M_start + (M_end - M_start) * t
        new_temp = temp_start + (temp_end - temp_start) * t
        new_params = {
            'M': jnp.array([new_M]),
            'temp': jnp.array([new_temp]),
        }
        return new_params


# https://github.com/take-koshizuka/NLSB/blob/main/models/potential_2d.py
class BoxPotential(LagrangianPotentialBase):
    xmin: float = -0.5
    xmax: float = 0.5
    ymin: float = -0.5
    ymax: float = 0.5

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux = (nn.sigmoid((x[0] - self.xmin) / self.temp) - \
              nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (nn.sigmoid((x[1] - self.ymin) / self.temp) - \
              nn.sigmoid((x[1] - self.ymax) / self.temp))
        U = -Ux * Uy
        return self.M*U


class SlitPotential(LagrangianPotentialBase):
    xmin: float = -0.1
    xmax: float = 0.1
    ymin: float = -0.25
    ymax: float = 0.25
    M_bounds = (0., 1.)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux = (nn.sigmoid((x[0] - self.xmin) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (nn.sigmoid((x[1] - self.ymin) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax) / self.temp)) - 1.
        U = Ux * Uy
        return self.M*U

class BabyMazePotential(LagrangianPotentialBase):
    xmin1: float = -0.5
    xmax1: float = -0.3
    ymin1: float = -1.99
    ymax1: float = -0.15
    xmin2: float = 0.3
    xmax2: float = 0.5
    ymin2: float = 0.15
    ymax2: float = 1.99
    M_bounds = (0., 10.)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux1 = (nn.sigmoid((x[0] - self.xmin1) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax1) / self.temp))
        Ux2 = (nn.sigmoid((x[0] - self.xmin2) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax2) / self.temp))

        Uy1 = (nn.sigmoid((x[1] - self.ymin1) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax1) / self.temp)) - 1.

        Uy2 = (nn.sigmoid((x[1] - self.ymin2) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax2) / self.temp)) - 1.
        U = Ux1 * Uy1 + Ux2 * Uy2
        return self.M*U

@dataclass
class WellPotential(LagrangianPotentialBase):
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        U = -jnp.sum(x**2)
        return self.M*U

@dataclass
class HillPotential(LagrangianPotentialBase):
    M_bounds = (0., 0.05)
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        U = -jnp.exp(-jnp.sum(x**2))
        return self.M*U


@dataclass
class GSB_GMM_Potential(LagrangianPotentialBase):
    centers = jnp.array([[6,6], [6,-6], [-6,-6]])
    radius = 1.5
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        V = 0.
        for i in range(self.centers.shape[0]):
            dist = jnp.linalg.norm(x - self.centers[i])
            V -= self.M * nn.sigmoid((self.radius - dist) / self.temp)
        return V


@dataclass
class GSB_VNeck_Potential(LagrangianPotentialBase):
    c_sq = 0.36
    coef = 5
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        xs_sq = x*x
        d = self.coef * xs_sq[0] - xs_sq[1]

        return - self.M * nn.sigmoid((-self.c_sq - d) / self.temp)


@dataclass
class GSB_STunnel_Potential(LagrangianPotentialBase):
    a, b, c = 20, 1, 90
    centers = [[5,6], [-5,-6]]
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        V = 0.0
        d = self.a*(x[0]-self.centers[0][0])**2 + \
            self.b*(x[1]-self.centers[0][1])**2
        V -= self.M * nn.sigmoid((self.c - d) / self.temp)

        d = self.a*(x[0]-self.centers[1][0])**2 + \
            self.b*(x[1]-self.centers[1][1])**2
        V -= self.M * nn.sigmoid((self.c - d) / self.temp)

        return V
