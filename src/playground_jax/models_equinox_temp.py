#TODO!
class DenseNN(eqx.Module):
    """
    A Multilayer perceptor model with interlayer connections.
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
            self.layers += [eqx.nn.Linear(sum(sizes[:j+1]), sizes[j+1], key=subkeys[j]), act]
            #nn.Linear(int(np.sum(self.d_layers[:i+1])), self.d_layers[i+1], bias=True),

    def __call__(self, x):
        for layer in self.layers[:-1]:
            z = layer(x)
            x = jnp.hstack((x, z))
        return self.layers[-1](x)

class DenseNN_4layers(eqx.Module):
    layers: list

    def __init__(self, d_in, d_out, d_hidden, key):
        n_layers = 4
        subkeys = random.split(key, n_layers)

        # the layers contain trainable parameters.
        self.layers = [eqx.nn.Linear(d_in, d_hidden[0], key=subkeys[0]),
                       eqx.nn.Linear(d_hidden[0], d_hidden[1], key=subkeys[1]),
                       eqx.nn.Linear(d_hidden[1], d_hidden[2], key=subkeys[2]),
                       eqx.nn.Linear(d_hidden[2], d_out, key=subkeys[3])]


    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.cat([x, nn.tanh(layer(x))], dim=1)
        #x = nn.softplus(self.layers[-1](x))
        x = self.layers[-1](x)
        return x

