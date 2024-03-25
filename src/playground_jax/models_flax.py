from typing import Sequence

from jax import numpy as jnp
from flax import linen as nn

from jax_utils import Array, f32
from gaussian_utils import mvn_pdf_basis_diagonal

class MLP(nn.Module):
    """
    A simple Multilayer perceptor model.
    """

    features: Sequence[int]
    act_fn : callable
    final_act_fn : callable

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.act_fn(nn.Dense(feat)(x))
        x = self.final_act_fn(nn.Dense(self.features[-1])(x))
        return x

class MLP_basic(nn.Module):

    d_out: int

    @nn.compact
    def __call__(self, x):
        for i in range(3):
            x = nn.tanh(nn.Dense(32)(x))
        x = nn.Dense(self.d_out)(x)
        return x

class DenseNN(nn.Module):
    """
    A Multilayer perceptor model with interlayer connections.
    """

    features: Sequence[int]
    act_fn : callable
    final_act_fn : callable

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            z = self.act_fn(nn.Dense(feat)(x))
            x = jnp.hstack((x, z))
        x = self.final_act_fn(nn.Dense(self.features[-1])(x))
        return x
