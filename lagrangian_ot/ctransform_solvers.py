# Copyright (c) Meta Platforms, Inc. and affiliates

import abc
from typing import Callable, Literal, NamedTuple, Optional
import dataclasses
import collections

import jax
import jax.numpy as jnp
import optax
import flax
from jaxopt import LBFGS

from ott import utils
from lagrangian_ot import geometries


class CTransformResults(NamedTuple):
    val: float
    solution: jnp.ndarray
    num_iter: int


class CTransformSolver(abc.ABC):
    geometry: geometries.GeometryBase

    @abc.abstractmethod
    def solve(
        self,
        f: Callable[[jnp.ndarray], jnp.ndarray],
        source: jnp.ndarray,
        target_init: Optional[jnp.ndarray] = None
    ) -> CTransformResults:
        pass

@dataclasses.dataclass
class CTransformLBFGS(CTransformSolver):
    gtol: float = 1e-3
    max_iter: int = 10
    max_linesearch_iter: int = 10
    linesearch_type: Literal["zoom", "backtracking"] = "backtracking"
    decrease_factor: float = 0.66
    ls_method: Literal["wolf", "strong-wolfe"] = "strong-wolfe"

    def solve(
        self,
        geometry_params: flax.core.FrozenDict,
        f: Callable[[jnp.ndarray], jnp.ndarray],
        source: jnp.ndarray,
        target_init: Optional[jnp.array] = None
    ) -> CTransformResults:
        assert source.ndim == 1

        cost_fn = lambda target: self.geometry.apply(
            {'params': geometry_params},
            source, target, method=self.geometry.cost)

        def objective(target):
            target = self.geometry.project(target)
            return cost_fn(target) - f(target)

        solver = LBFGS(
            fun=objective,
            tol=self.gtol,
            maxiter=self.max_iter,
            decrease_factor=self.decrease_factor,
            linesearch=self.linesearch_type,
            condition=self.ls_method,
            implicit_diff=False,
            unroll=False
        )

        out = solver.run(source if target_init is None else target_init)
        solution = self.geometry.project(out.params)
        return CTransformResults(
            val=out.state.value, solution=solution, num_iter=out.state.iter_num
        )

@dataclasses.dataclass
class CTransformAdam(CTransformSolver):
    gtol: float = 1e-3
    max_iter: int = 10

    adam_kwargs: Optional[dict] = None
    lr_schedule_kwargs: Optional[dict] = None

    def __post_init__(self):
        if self.adam_kwargs is None:
            self.adam_kwargs = {'b1': 0.9, 'b2': 0.999}
        if self.lr_schedule_kwargs is None:
            self.lr_schedule_kwargs = {
                'init_value': 0.1,
                'decay_steps': self.max_iter,
                'alpha': 1e-4,
            }


    def solve(
        self,
        geometry_params: flax.core.FrozenDict,
        f: Callable[[jnp.ndarray], jnp.ndarray],
        source: jnp.ndarray,
        target_init: Optional[jnp.array] = None
    ) -> CTransformResults:
        assert source.ndim == 1

        cost_fn = lambda target: self.geometry.apply(
            {'params': geometry_params},
            source, target, method=self.geometry.cost)

        def objective(target):
            return cost_fn(target) - f(target)

        lr_schedule = optax.cosine_decay_schedule(
            **self.lr_schedule_kwargs)
        adam = optax.adam(learning_rate=lr_schedule, **self.adam_kwargs)
        opt_state = adam.init(target_init)

        obj, grad = jax.value_and_grad(objective)(target_init)

        LoopState = collections.namedtuple("LoopState", "i x grad opt_state obj")
        init_state = LoopState(0, target_init, grad, opt_state, obj)

        def cond_fun(state):
            return (state.i < self.max_iter) & \
                (jnp.linalg.norm(state.grad, ord=jnp.inf) > self.gtol)

        def body_fun(state):
            updates, new_opt_state = adam.update(state.grad, state.opt_state, state.x)
            x_new = optax.apply_updates(state.x, updates)
            x_new = self.geometry.project(x_new)
            obj_new, grad_new = jax.value_and_grad(objective)(x_new)
            return LoopState(
                state.i+1, x_new, grad_new, new_opt_state, obj_new)

        state = jax.lax.while_loop(cond_fun, body_fun, init_state)

        obj = state.obj
        x = state.x
        n_iter = state.i
        return CTransformResults(
            val=obj, solution=x, num_iter=n_iter,
        )



# DEFAULT_CTRANSFORM_SOLVER = CTransformLBFGS(
#     gtol=1e-5,
#     max_iter=20,
#     max_linesearch_iter=20,
#     linesearch_type="backtracking",
# )

DEFAULT_CTRANSFORM_SOLVER = CTransformAdam(
    gtol=1e-5,
    max_iter=20,
)
