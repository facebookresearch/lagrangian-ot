# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
import functools

from dataclasses import dataclass

from abc import ABC, abstractmethod

import copy
from typing import Callable, Optional, Tuple, Dict


from enum import Enum

from lagrangian_ot import (
    metrics,
    geodesics,
    lagrangian_potentials,
    splines,
    spline_amortizer,
)


class DistanceModes:
    GEODESIC = "geodesic"
    SQUARED_GEODESIC = "sq_geodesic"
    LAGRANGIAN = "lagrangian"


def get(name, geometry_kwargs):
    if name == "sq_euclidean":
        return SqEuclidean()
    elif name == "gmm":
        return SqEuclidean(bounds=(-20, 20))
    elif name == "sq_euclidean_manifold":
        return MetricManifold(
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.EuclideanMetric,
            **geometry_kwargs,
        )
    elif name == "scarvelis_circle":
        return MetricManifold(
            bounds=(-1.5, 1.5),
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.CircleMetric,
            **geometry_kwargs,
        )
    elif name == "scarvelis_vee":
        xbounds = (-2.5, 15)
        ybounds = (-15, 15)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
        return MetricManifold(
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.VeeMetric,
            xbounds=xbounds,
            ybounds=ybounds,
            bounds=bounds,
            **geometry_kwargs,
        )
    elif name == "scarvelis_xpath":
        return MetricManifold(
            bounds=(-1.5, 1.5),
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.XMetric,
            **geometry_kwargs,
        )
    elif name == "lsb_box":
        return MetricManifold(
            bounds=(-1.5, 1.5),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.BoxPotential,
            **geometry_kwargs,
        )
    elif name == "lsb_slit":
        return MetricManifold(
            bounds=(-1.5, 1.5),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.SlitPotential,
            **geometry_kwargs,
        )
    elif name == "lsb_hill":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.HillPotential,
            **geometry_kwargs,
        )
    elif name == "lsb_well":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.WellPotential,
            **geometry_kwargs,
        )
    elif name == "gsb_gmm":
        return MetricManifold(
            bounds=(-20, 20),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.GSB_GMM_Potential,
            **geometry_kwargs,
        )
    elif name == "gsb_vneck":
        xbounds = (-10, 10)
        ybounds = (-8, 8)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
        return MetricManifold(
            xbounds=xbounds,
            ybounds=ybounds,
            bounds=bounds,
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.GSB_VNeck_Potential,
            **geometry_kwargs,
        )
    elif name == "gsb_stunnel":
        xbounds = (-15, 15)
        ybounds = (-7.5, 7.5)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
        return MetricManifold(
            xbounds=xbounds,
            ybounds=ybounds,
            bounds=bounds,
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.GSB_STunnel_Potential,
            **geometry_kwargs,
        )
    elif name == "babymaze":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.LAGRANGIAN,
            metric_initializer_fn=metrics.EuclideanMetric,
            lagrangian_potential_initializer_fn=lagrangian_potentials.BabyMazePotential,
            **geometry_kwargs,
        )
    elif name == "neural_net_metric":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.NeuralNetMetric,
            **geometry_kwargs,
        )
    else:
        raise ValueError(f"Unknown geometry: {name}")


@dataclass
class GeometryBase(ABC, nn.Module):
    D: int = 2  # dimension of the ambient space
    bounds: Tuple = (-2, 2)  # bounds of the measures

    # for 2d geometries
    xbounds: Tuple = None
    ybounds: Tuple = None

    @abstractmethod
    def cost(self, x, y):
        pass

    @abstractmethod
    def path(self, x, y, num_points=20):
        pass

    @abstractmethod
    def project(self, x):
        pass

    def add_plot_background(self, params, ax, xlims, ylims=None, **kwargs):
        pass


eps = 1e-5  # TODO: Other stabilization?
divsin = lambda x: x / jnp.sin(x)
sindiv = lambda x: jnp.sin(x) / (x + eps)
divsinh = lambda x: x / jnp.sinh(x)
sinhdiv = lambda x: jnp.sinh(x) / (x + eps)


class Sphere(GeometryBase):
    jitter: float = 1e-2

    def exponential_map(self, x, v):
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        return x * jnp.cos(v_norm) + v * sindiv(v_norm)

    def log(self, x, y):
        xy = (x * y).sum(axis=-1, keepdims=True)
        xy = jnp.clip(xy, a_min=-1 + 1e-6, a_max=1 - 1e-6)
        val = jnp.arccos(xy)
        return divsin(val) * (y - xy * x)

    def tangent_projection(self, x, u):
        proj_u = u - x * x.dot(u)
        return proj_u

    def tangent_orthonormal_basis(self, x, dF):
        assert x.ndim == 2

        if x.shape[1] == 2:
            E = x[:, jnp.array([1, 0])] * jnp.array([-1.0, 1.0])
            E = E.reshape(*E.shape, 1)
        elif x.shape[1] == 3:
            norm_v = dF / jnp.linalg.norm(dF, axis=-1, keepdims=True)
            E = jnp.dstack([norm_v, jnp.cross(x, norm_v)])
        else:
            raise NotImplementedError()

        return E

    def dist(self, x, y):
        inner = jnp.matmul(x, y)
        inner = inner / (1 + self.jitter)
        return jnp.arccos(inner)

    ### made cost = dist^2
    def cost(self, x, y):
        return self.dist(x, y) ** 2 / 2.0

    ##changed to project
    def project(self, x):
        x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x

    def transp(self, x, y, u):
        yu = jnp.sum(y * u, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        return u - yu / (1 + xy) * (x + y)

    def logdetexp(self, x, u):
        norm_u = jnp.linalg.norm(u, axis=-1)
        val = jnp.log(jnp.abs(sindiv(norm_u)))
        return (u.shape[-1] - 2) * val

    def zero(self):
        y = jnp.zeros(self.D)
        y = y.at[..., 0].set(-1.0)
        return y

    def zero_like(self, x):
        y = jnp.zeros_like(x)
        y = y.at[..., 0].set(-1.0)
        return y

    def squeeze_tangent(self, x):
        return x[..., 1:]

    def unsqueeze_tangent(self, x):
        return jnp.concatenate((jnp.zeros_like(x[..., :1]), x), axis=-1)

    # TODO: code up the path
    def path(self, x, y, num_points=20):
        pass


class SqEuclidean(GeometryBase):
    def cost(self, x, y):
        assert x.ndim == 1 and y.ndim == 1
        return 0.5 * jnp.linalg.norm(x - y) ** 2

    def path(self, x, y, num_points=20):
        assert x.ndim == 1 and y.ndim == 1
        return jnp.linspace(x, y, num_points)

    def project(self, x):
        return x


@dataclass
class MetricManifold(GeometryBase):
    distance_mode: DistanceModes = DistanceModes.SQUARED_GEODESIC
    metric_initializer_fn: Callable = metrics.EuclideanMetric
    spline_model_initializer_fn: Callable = spline_amortizer.SplineMLP
    lagrangian_potential_initializer_fn: Optional[Callable] = None
    spline_solver_kwargs: Optional[Dict] = None

    def __post_init__(self):
        if self.spline_solver_kwargs is None:
            self.spline_solver_kwargs = {}
        self.spline_geodesic_solver = geodesics.SplineSolver(
            D=self.D, **self.spline_solver_kwargs
        )
        self.spline_amortizer = spline_amortizer.SplineAmortizer(
            self, self.spline_geodesic_solver
        )
        super().__post_init__()

    def setup(self):
        self.metric_module = self.metric_initializer_fn()
        if self.lagrangian_potential_initializer_fn is not None:
            self.lagrangian_potential_module = (
                self.lagrangian_potential_initializer_fn()
            )

        self.spline_model = self.spline_model_initializer_fn(
            self.spline_geodesic_solver.num_spline_params
        )

    def predict_spline_params(self, x, y):
        return self.spline_model(x, y)

    def metric(self, x):
        return self.metric_module(x)

    def lagrangian_potential(self, x):
        return self.lagrangian_potential_module(x)

    def path(self, x, y, num_points=20):
        assert x.ndim == 1 and y.ndim == 1
        init_spline_params = jax.lax.stop_gradient(self.predict_spline_params(x, y))

        initializing = self.is_mutable_collection("params")
        if initializing:
            # don't improve the solution
            # (linen doesn't work well with jax.lax.while_loop)
            ts = jnp.linspace(0, 1, num_points)
            xs = splines.compute_spline(
                x=x,
                y=y,
                basis=self.spline_amortizer.basis,
                params=init_spline_params,
                ts=ts,
            )
            E = self.curve_energy(xs)
            return xs

        out = self.spline_geodesic_solver.solve(
            self.curve_energy,
            x,
            y,
            init_params=init_spline_params,
            num_final_points=num_points,
        )
        return out.mu

    def energy_at_point(self, x, v):
        M = self.metric(x)
        if (
            self.distance_mode == DistanceModes.GEODESIC
            or self.distance_mode == DistanceModes.SQUARED_GEODESIC
        ):
            kinetic = jnp.sqrt(v @ M @ v)
        else:
            kinetic = 0.5 * v @ M @ v
        lagrangian_potential = (
            self.lagrangian_potential(x)
            if self.lagrangian_potential_initializer_fn is not None
            else 0.0
        )
        return kinetic - lagrangian_potential

    def curve_energy(self, xs):
        assert xs.ndim == 2
        T = xs.shape[0]
        ds = (xs[1:] - xs[:-1]) + 1e-6
        Es = jax.vmap(self.energy_at_point)(xs[:-1], ds)
        return Es.sum()

    def cost(self, x, y):
        assert x.ndim == 1 and y.ndim == 1
        gamma = self.path(x, y)
        num_points = gamma.shape[0]
        E = self.curve_energy(gamma)
        if self.distance_mode == DistanceModes.SQUARED_GEODESIC:
            E = 0.5 * E**2
        return E

    def project(self, x):
        return jnp.clip(x, *self.bounds)

    def add_plot_background(self, params, axs, xlims, ylims=None, alpha=1.0):
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        if issubclass(
            self.metric_initializer_fn, metrics.ScarvelisMetric
        ) or issubclass(self.metric_initializer_fn, metrics.NeuralNetMetric):
            grid_size = 21

            assert len(xlims) == 2
            if ylims is None:
                ylims = xlims

            if not hasattr(self, "eigs_vmap_jit"):

                @functools.partial(jax.vmap, in_axes=(None, 0))
                def eigs_vmap(params, x):
                    A = self.apply({"params": params}, x, method=self.metric)
                    vals, vecs = jnp.linalg.eigh(A)
                    return vals, vecs.T

                self.eigs_vmap_jit = jax.jit(eigs_vmap)

            xflat, x1, x2 = _get_grid(xlims, ylims, grid_size)
            vals, vecs = self.eigs_vmap_jit(params, xflat)

            u = vecs[:, 0, 0].reshape(x1.shape)
            v = vecs[:, 0, 1].reshape(x1.shape)

            for ax in axs:
                ax.quiver(x1.ravel(), x2.ravel(), u, v, alpha=alpha)
                ax.quiver(x1.ravel(), x2.ravel(), -u, -v, alpha=alpha)

                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)
        elif self.lagrangian_potential_initializer_fn is not None:
            grid_size = 201
            assert len(xlims) == 2
            if ylims is None:
                ylims = xlims

            if not hasattr(self, "lagrangian_potential_vmap_jit"):

                @functools.partial(jax.vmap, in_axes=(None, 0))
                def lagrangian_potential_vmap(params, x):
                    return self.apply(
                        {"params": params}, x, method=self.lagrangian_potential
                    )

                self.lagrangian_potential_vmap_jit = jax.jit(lagrangian_potential_vmap)

            xflat, x1, x2 = _get_grid(xlims, ylims, grid_size)
            vals = -self.lagrangian_potential_vmap_jit(params, xflat).reshape(x1.shape)

            for ax in axs:
                CS = ax.contourf(x1, x2, vals, cmap="Blues")
                # fig = ax.get_figure()
                # fig.colorbar(CS, ax=ax)

                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)

    def __hash__(self):
        # not perfect, for passing as a static arg to jax.jit
        return hash(
            (
                self.distance_mode,
                self.metric_initializer_fn,
                self.spline_model_initializer_fn,
                self.lagrangian_potential_initializer_fn,
            )
        )


def _get_grid(xlims: Tuple[float, float], ylims: Tuple[float, float], grid_size=21):
    xs = np.linspace(*xlims, num=grid_size)
    ys = np.linspace(*ylims, num=grid_size)
    x1, x2 = np.meshgrid(xs, ys)
    x1flat = x1.ravel()
    x2flat = x2.ravel()
    xflat = np.stack((x1flat, x2flat)).T
    return xflat, x1, x2
