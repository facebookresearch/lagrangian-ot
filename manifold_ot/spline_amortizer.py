# Copyright (c) Meta Platforms, Inc. and affiliates

import jax
import jax.numpy as jnp
import optax

import flax
from flax import linen as nn

from typing import Any

from dataclasses import dataclass

from . import splines, meters, geodesics

class SplineMLP(nn.Module):
    out_dims: int
    num_hidden: int = 1024

    @nn.compact
    def __call__(self, x, y):
        # TODO: parameterize, make deeper
        squeeze = x.ndim == 1
        if squeeze:
            x = jnp.expand_dims(x, 0)
            y = jnp.expand_dims(y, 0)
        assert x.ndim == 2
        z = jnp.concatenate([x, y], axis=1)
        z = nn.relu(nn.Dense(self.num_hidden)(z))
        z = nn.relu(nn.Dense(self.num_hidden)(z))
        z = nn.Dense(self.out_dims)(z)
        if squeeze:
            z = jnp.squeeze(z, 0)
        return z


@dataclass
class SplineAmortizer:
    geometry: "GeometryBase"
    spline_geodesic_solver: geodesics.SplineSolver

    def __post_init__(self):
        self.D = self.spline_geodesic_solver.D
        self.num_params_spline = self.spline_geodesic_solver.spline_basis.shape[-1] * self.D
        self.basis = self.spline_geodesic_solver.spline_basis

        grad_clip = 1.0 # TODO
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(1e-4),
        )
        self.opt_state = None


    def loss_fn(self, params_geometry, xs, ys):
        predict_spline_fn = lambda params_geometry, x, y: self.geometry.apply(
            {'params': params_geometry}, x, y,
            method=self.geometry.predict_spline_params)

        curve_energy = lambda xs: self.geometry.apply(
            {'params': params_geometry}, xs,
            method=self.geometry.curve_energy)

        def spline_energy(params_spline, x, y):
            assert x.ndim == 1 and y.ndim == 1

            # TODO: num, also set spline knots elsewhere?
            ts = jnp.linspace(0., 1., num=20)

            path = splines.compute_spline(
                x=x, y=y, basis=self.basis, params=params_spline, ts=ts)
            E = curve_energy(path)
            return E

        spline_energy_vmap = jax.vmap(spline_energy)

        params_spline = predict_spline_fn(params_geometry, xs, ys)
        Es = spline_energy_vmap(params_spline, xs, ys)
        return jnp.mean(Es)


    def update_fn(self, params_geometry, opt_state, xs, ys):
        loss_value_and_grad = jax.value_and_grad(self.loss_fn, argnums=0)
        loss, grads = loss_value_and_grad(params_geometry, xs, ys)
        # TODO: this is inefficiently computing all gradients w.r.t.
        # the geometry but only using the spline ones, there may be
        # a better way to do this
        grads = grads['spline_model']
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params_spline_model = optax.apply_updates(
            params_geometry['spline_model'], updates)
        # params_geometry = params_geometry.unfreeze()
        params_geometry['spline_model'] = new_params_spline_model
        grad_norm = jnp.linalg.norm(jax.tree_util.tree_flatten(grads)[0][0])
        return params_geometry, opt_state, loss, grad_norm

    def train_single(self, params_geometry, source_samples, target_samples, verbose=True):
        assert self.opt_state is not None
        if verbose:
            print('fitting spline amortizer (single step)')
        params_geometry = params_geometry.unfreeze()
        params_geometry, self.opt_state, loss, grad_norm = self.update_fn_jit(
            params_geometry, self.opt_state, source_samples, target_samples)
        params_geometry = flax.core.freeze(params_geometry)
        if verbose:
            print(f'  loss: {loss:.2e}')
        return params_geometry

    def train(self, params_geometry,
              source_sampler, target_sampler, max_iter,
              grad_norm_threshold=None, callback=None):
        print('fitting spline amortizer')
        params_geometry = params_geometry.unfreeze()

        if self.opt_state is None:
            params_spline_model = params_geometry['spline_model']
            self.opt_state = self.optimizer.init(params_spline_model)
            self.update_fn_jit = jax.jit(self.update_fn)

        loss_meter = meters.EMAMeter(0.9)
        loss_grad_meter = meters.EMAMeter(0.9)

        for i in range(max_iter):
            xs = next(source_sampler)
            ys = next(target_sampler)
            params_geometry, self.opt_state, loss, grad_norm = self.update_fn_jit(
                params_geometry, self.opt_state, xs, ys)
            loss_meter.update(loss)
            loss_grad_meter.update(grad_norm)
            if i % 1000 == 0:
                print(f'[{i}] loss: {loss_meter.value:.2e} grad_norm: {loss_grad_meter.value:.2e}')
            if grad_norm_threshold is not None and loss_grad_meter.value < grad_norm_threshold:
                break

            if callback is not None:
                callback(i)

        params_geometry = flax.core.freeze(params_geometry)
        return params_geometry
