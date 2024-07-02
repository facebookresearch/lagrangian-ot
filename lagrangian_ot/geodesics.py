# Copyright (c) Meta Platforms, Inc. and affiliates

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import optimize
import jaxopt
import optax

from collections import namedtuple

from dataclasses import dataclass
from abc import ABC, abstractmethod

from lagrangian_ot import splines

GPState = namedtuple("GPState", "i ddc mu dmu cost")
SplineState = namedtuple("SplineState", "i params opt_state E grad_norm")
SolverOut = namedtuple("SolverOut", "mu dmu num_iter cost")

@dataclass
class SolverBase(ABC):
    D: int

    @abstractmethod
    def solve(self, geo_eq, x0, x1):
        pass


@dataclass
class SplineSolver(SolverBase):
    num_spline_nodes: int = 20
    grad_tol: float = 1e-5
    init_lr: float = 1e-2
    num_spline_points_eval: int = 21
    max_iter: int = 20

    def __post_init__(self):
        self.spline_basis = splines.get_basis(
            self.D, num_nodes=self.num_spline_nodes)
        self.num_spline_params = self.spline_basis.shape[-1] * self.D

    def solve(self, energy_fn, x0, x1, init_params, num_final_points=None):
        ts = jnp.linspace(0., 1., num=self.num_spline_points_eval)
        if self.max_iter == 0:
            if num_final_points is not None:
                ts = jnp.linspace(0., 1., num=num_final_points)
            xs = splines.compute_spline(
                x=x0, y=x1, basis=self.spline_basis, params=init_params, ts=ts)
            E = energy_fn(xs)
            return SolverOut(mu=xs, dmu=None, num_iter=0, cost=E)

        def F(params):
            xs = splines.compute_spline(
                x=x0, y=x1, basis=self.spline_basis, params=params, ts=ts)
            E = energy_fn(xs)
            return E

        def cond_fun(spline_solver_state):
            return (spline_solver_state.i < self.max_iter) & \
                (spline_solver_state.grad_norm > self.grad_tol)

        def body_fun(spline_solver_state):
            E, g = jax.value_and_grad(F)(spline_solver_state.params)
            updates, opt_state = opt.update(g, spline_solver_state.opt_state)
            params = optax.apply_updates(spline_solver_state.params, updates)
            return SplineState(
                i=spline_solver_state.i+1,
                params=params, opt_state=opt_state,
                E=E, grad_norm=jnp.linalg.norm(g))

        lr_schedule = optax.cosine_decay_schedule(
            self.init_lr, decay_steps=self.max_iter, alpha=1e-1)
        opt = optax.adam(lr_schedule, b1=0.9, b2=0.999)
        opt_state = opt.init(init_params)
        spline_solver_state = SplineState(
            i=0, params=init_params, opt_state=opt_state,
            E=jnp.inf, grad_norm=jnp.inf)
        spline_solver_state = lax.stop_gradient(
            lax.while_loop(cond_fun, body_fun, spline_solver_state))
        params = spline_solver_state.params
        num_iter = spline_solver_state.i
        E = spline_solver_state.E

        if num_final_points is not None:
            ts = jnp.linspace(0., 1., num=num_final_points)
        xs = splines.compute_spline(
            x=x0, y=x1, basis=self.spline_basis, params=params, ts=ts)
        return SolverOut(mu=xs, dmu=None, num_iter=num_iter, cost=E)
