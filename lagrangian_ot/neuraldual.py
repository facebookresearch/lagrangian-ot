# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Sequence,
)

import functools
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

import collections

import flax
from flax import core
from flax.core import frozen_dict
from flax.training import train_state

from copy import copy

from ott.geometry import costs

import matplotlib as mpl
import matplotlib.pyplot as plt

from lagrangian_ot import ctransform_solvers, models, geometries

Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable
Conj_t = Optional[ctransform_solvers.CTransformSolver]
Potential_t = Callable[[jnp.ndarray], float]

Info = collections.namedtuple("Info", "dual_loss amor_loss num_ctransform_iter target_hat")
UpdateOut = collections.namedtuple("UpdateOut", "states info")


class ManifoldW2NeuralDual:
    def __init__(
            self,
            geometry: geometries.GeometryBase,
            target_potential: models.ModelBase,
            source_map: models.ModelBase,
            ctransform_solver: Conj_t = ctransform_solvers.DEFAULT_CTRANSFORM_SOLVER,
            amortization_loss: Literal["objective", "regression"] = "regression",
    ):
        self.geometry = geometry
        self.amortization_loss = amortization_loss

        self.ctransform_solver = ctransform_solver
        self.ctransform_solver.geometry = geometry

        self.target_potential = target_potential
        self.source_map = source_map

        self.target_potential_apply_jit = jax.jit(self.target_potential.apply)
        self.source_map_apply_jit = jax.jit(self.source_map.apply)

        self.finetune_target_hat_vmap_jit = jax.jit(jax.vmap(
            lambda params_target, params_geometry, source, target_init: self.ctransform_solver.solve(
                params_geometry,
                functools.partial(self.target_potential.apply, {'params': params_target}),
                source, target_init=target_init
            ), in_axes=(None, None, 0, 0)))

        self.pushforward = lambda params_source, params_target, params_geometry, source: self.ctransform_solver.solve(
            params_geometry,
            functools.partial(self.target_potential.apply, {'params': params_target}),
            source,
            target_init=self.source_map.apply({'params': params_source}, source)
        )
        self.pushforward_jit = jax.jit(self.pushforward)
        self.pushforward_jit_vmap = jax.jit(jax.vmap(self.pushforward, in_axes=(None, None, None, 0)))

        self.path_jit = jax.jit(
            lambda params_geometry, x, y: self.geometry.apply(
                {'params': params_geometry}, x, y, method=self.geometry.path))
        self.path_jit_vmap = jax.jit(jax.vmap(
            lambda params_geometry, x, y: self.geometry.apply(
                {'params': params_geometry}, x, y, method=self.geometry.path),
            in_axes=(None, 0, 0)))

        self.update_fn_jit = jax.jit(self.update_fn)

    def initialize_states(
            self, optimizer_target_potential, optimizer_source_map,
            key, source_samples, target_samples):
        key, key1, key2 = jax.random.split(key, 3)
        params_target_potential = self.target_potential.init(key1, target_samples)['params']
        params_source_map = self.source_map.init(key2, source_samples)['params']

        state_target_potential = train_state.TrainState.create(
            apply_fn=self.target_potential.apply,
            params=params_target_potential, tx=optimizer_target_potential
        )
        state_source_map = train_state.TrainState.create(
            apply_fn=self.source_map.apply,
            params=params_source_map, tx=optimizer_source_map,
        )
        return state_target_potential, state_source_map


    def state_from_dicts(self, optimizer, net, state_dicts):
        params = state_dicts['params']
        state = train_state.TrainState.create(
            apply_fn=net.apply, params=params, tx=optimizer
        )
        state = flax.serialization.from_state_dict(state, state_dicts)
        return state


    def state_to_dicts(self, state):
        state_dict = flax.serialization.to_state_dict(state)
        return state_dict


    def loss_fn(self, params_target_potential, params_source_map, params_geometry, batch):
        """Loss function for both potentials."""
        source, target = batch["source"], batch["target"]

        init_target_hat = self.source_map.apply({'params': params_source_map}, source)

        target_potential_partial = functools.partial(
            self.target_potential.apply, {'params': params_target_potential})
        if self.ctransform_solver is not None:
            finetune_target_hat = lambda source, target_init: self.ctransform_solver.solve(
                params_geometry, target_potential_partial, source, target_init=target_init
            )
            finetune_target_hat = jax.vmap(finetune_target_hat)
            out = finetune_target_hat(source, init_target_hat)
            target_hat_detach = jax.lax.stop_gradient(out.solution)
            num_ctransform_iter = jnp.mean(out.num_iter)
            if hasattr(self.geometry, 'bounds'):
                target_hat_detach = jnp.clip(target_hat_detach, *self.geometry.bounds)
        else:
            target_hat_detach = init_target_hat
            num_ctransform_iter = 0

        target_potential = target_potential_partial(target)
        cost_vmap = jax.vmap(lambda x, y: self.geometry.apply(
            {'params': params_geometry}, x, y, method=self.geometry.cost))
        source_potential = cost_vmap(source, target_hat_detach) - \
            target_potential_partial(target_hat_detach)
        dual_source = source_potential.mean()
        dual_target = target_potential.mean()
        dual_loss = -dual_source - dual_target

        if self.amortization_loss == "regression":
            amor_loss = ((init_target_hat - target_hat_detach) ** 2).mean()
        elif self.amortization_loss == "objective":
            import ipdb; ipdb.set_trace()
            batch_dot = jax.vmap(jnp.dot)
            f_value_parameters_detached = lambda x: f_value(
                    jax.lax.stop_gradient(params_f), x
            )
            amor_loss = (
                    f_value_parameters_detached(init_source_hat) -
                    batch_dot(init_source_hat, target)
            ).mean()
        else:
            raise ValueError("Amortization loss has been misspecified.")

        loss = dual_loss + amor_loss
        return loss, Info(dual_loss, amor_loss, num_ctransform_iter, target_hat_detach)


    def update_fn(self, state_target_potential, state_source_map, params_geometry, batch):
        """Step function of either training or validation."""
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=[0, 1], has_aux=True)
        (loss, info), (grads_target_potential, grads_source_map) = grad_fn(
                state_target_potential.params,
                state_source_map.params,
                params_geometry,
                batch,
        )
        new_states = (
            state_target_potential.apply_gradients(grads=grads_target_potential),
            state_source_map.apply_gradients(grads=grads_source_map)
        )
        return UpdateOut(new_states, info)


    def plot_forward_map(
            self,
            source: jnp.ndarray,
            target: jnp.ndarray,
            state_source_map: train_state.TrainState,
            state_target_potential: train_state.TrainState,
            params_geometry: flax.core.FrozenDict,
            ax: Optional["plt.Axes"] = None,
            legend: bool = True,
            scatter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["plt.Figure", "plt.Axes"]:
        if mpl is None:
            raise RuntimeError("Please install `matplotlib` first.")

        if scatter_kwargs is None:
            scatter_kwargs = {"alpha": 0.5}

        if ax is None:
            fig = plt.figure(facecolor="white")
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        # plot the source and target samples
        label_transport = r"transported"
        source_color, target_color = "#1A254B", "#A7BED3"

        ax.scatter(
                source[:, 0],
                source[:, 1],
                color=source_color,
                label="source",
                **scatter_kwargs,
        )
        ax.scatter(
                target[:, 0],
                target[:, 1],
                color=target_color,
                label="target",
                **scatter_kwargs,
        )

        # plot the transported samples
        base_samples = source
        N = base_samples.shape[0]

        transported_samples = self.pushforward_jit_vmap(
            state_source_map.params, state_target_potential.params, params_geometry, base_samples
        ).solution

        ax.scatter(
                transported_samples[:, 0],
                transported_samples[:, 1],
                color="#F2545B",
                label=label_transport,
                **scatter_kwargs,
        )
        paths = self.path_jit_vmap(params_geometry, base_samples, transported_samples)

        ax.plot(
            paths[:, :, 0].T,
            paths[:, :, 1].T,
            color=[0.5, 0.5, 1],
            alpha=0.3,
            lw=1,
        )

        if legend:
            legend_kwargs = {
                    "ncol": 3,
                    "loc": "upper center",
                    "bbox_to_anchor": (0.5, -0.05),
                    "edgecolor": "k"
            }
            ax.legend(**legend_kwargs)

        ax.set_title('source map')
        return fig, ax, transported_samples


    def plot_target_potential(
            self,
            source: jnp.ndarray,
            target: jnp.ndarray,
            state_target_potential: train_state.TrainState,
            quantile: float = 0.05,
            ax: Optional["mpl.axes.Axes"] = None,
            x_bounds: Tuple[float, float] = (-6, 6),
            y_bounds: Tuple[float, float] = (-6, 6),
            num_grid: int = 50,
            contourf_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["mpl.figure.Figure", "mpl.axes.Axes"]:
        r"""Plot the target potential.

        Args:
            quantile: quantile to filter the potentials with
            ax: axis to add the plot to
            num_grid: number of points to discretize the domain into a grid
                along each dimension
            contourf_kwargs: additional kwargs passed into
                :meth:`~matplotlib.axes.Axes.contourf`

        Returns:
            matplotlib figure and axis with the plots.
        """
        if contourf_kwargs is None:
            contourf_kwargs = {}

        ax_specified = ax is not None
        if not ax_specified:
            fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        else:
            fig = ax.get_figure()

        x1 = jnp.linspace(*x_bounds, num=num_grid)
        x2 = jnp.linspace(*y_bounds, num=num_grid)
        X1, X2 = jnp.meshgrid(x1, x2)
        X12flat = jnp.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
        apply_target_potential = jax.vmap(
            functools.partial(
                state_target_potential.apply_fn,
                {'params': state_target_potential.params}))
        Zflat = apply_target_potential(X12flat)
        Zflat = np.asarray(Zflat)
        vmin, vmax = np.quantile(Zflat, [quantile, 1. - quantile])
        Zflat = Zflat.clip(vmin, vmax)
        Z = Zflat.reshape(X1.shape)

        CS = ax.contourf(X1, X2, Z, cmap="Blues", **contourf_kwargs)

        source_color, target_color = "#1A254B", "#A7BED3"
        scatter_kwargs = {"alpha": 0.5}

        ax.scatter(
                source[:, 0],
                source[:, 1],
                color=source_color,
                label="source",
                **scatter_kwargs,
        )
        ax.scatter(
                target[:, 0],
                target[:, 1],
                color=target_color,
                label="target",
                **scatter_kwargs,
        )


        ax.set_xlim(*x_bounds)
        ax.set_ylim(*y_bounds)
        ax.set_title('target potential')
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1+pos.width*0.025, pos.y0, pos.width*0.05, pos.height])
        fig.colorbar(CS, cax=cax, orientation="vertical")

        if not ax_specified:
            fig.tight_layout()
        return fig, ax
