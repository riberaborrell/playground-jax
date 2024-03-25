from collections.abc import Callable

from jax import numpy as jnp, random, nn
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx

class Linear(eqx.Module):
    weight: Array
    bias: Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = random.split(key)
        self.weight = random.normal(wkey, (out_size, in_size))
        self.bias = random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    """
    A simple Multilayer perceptor model.
    """

    layers: list

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: list,
        activation: Callable=nn.relu,
        final_activation: Callable=lambda x: x,
        *,
        key: PRNGKeyArray,
    ):

        sizes = [d_in] + list(d_hidden) + [d_out]
        n_layers = len(sizes) - 1
        subkeys = random.split(key, n_layers)

        self.layers = []

        for j in range(n_layers):

            # actiavtion function
            act = activation if j < n_layers - 1 else final_activation

            # linear layer
            self.layers += [eqx.nn.Linear(sizes[j], sizes[j+1], key=subkeys[j]), act]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP_4layers(eqx.Module):
    layers: list

    def __init__(self, d_in, d_out, d_hidden, key):
        n_layers = 4
        subkeys = random.split(key, n_layers)

        # the layers contain trainable parameters.
        self.layers = [
            eqx.nn.Linear(d_in, d_hidden[0], key=subkeys[0]),
            eqx.nn.Linear(d_hidden[0], d_hidden[1], key=subkeys[1]),
            eqx.nn.Linear(d_hidden[1], d_hidden[2], key=subkeys[2]),
            eqx.nn.Linear(d_hidden[2], d_out, key=subkeys[3]),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.tanh(layer(x))
        return self.layers[-1](x)
