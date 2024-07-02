# Copyright (c) Meta Platforms, Inc. and affiliates

import abc
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state
from jax.nn import initializers

from ott.math import matrix_square_root

PotentialValueFn_t = Callable[[jnp.ndarray], jnp.ndarray]
PotentialGradientFn_t = Callable[[jnp.ndarray], jnp.ndarray]

class ModelBase(abc.ABC, nn.Module):
    """Base class for the neural solver models."""

    @property
    @abc.abstractmethod
    def is_potential(self) -> bool:
        """Indicates if the module defines the potential's value or the gradient.

        Returns:
            ``True`` if the module defines the potential's value, ``False``
            if it defines the gradient.
        """

class MLP(ModelBase):
    """A non-convex MLP.

    Args:
        dim_hidden: sequence specifying size of hidden dimensions. The output
            dimension of the last layer is automatically set to 1 if
            :attr:`is_potential` is ``True``, or the dimension of the input otherwise
        is_potential: Model the potential if ``True``, otherwise
            model the gradient of the potential
    """

    dim_hidden: Sequence[int]
    is_potential: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:    # noqa: D102
        squeeze = x.ndim == 1
        if squeeze:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2, x.ndim
        n_input = x.shape[-1]

        z = x
        for n_hidden in self.dim_hidden:
            Wx = nn.Dense(n_hidden, use_bias=True)
            z = nn.leaky_relu(Wx(z))

        if self.is_potential:
            Wx = nn.Dense(1, use_bias=True)
            z = Wx(z).squeeze(-1)
        else:
            Wx = nn.Dense(n_input, use_bias=True)
            z = x + Wx(z)

        return z.squeeze(0) if squeeze else z
