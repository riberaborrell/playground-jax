class MLP_isokann(nn.Module):

    d_out: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(32)(x))
        x = nn.relu(nn.Dense(16)(x))
        x = nn.relu(nn.Dense(8)(x))
        x = nn.softplus(nn.Dense(self.d_out)(x))
        return x

class GaussianAnsatz(nn.Module):
    means: Array
    sigma_i: Array

    @nn.compact
    def __call__(self, x):
        x = mvn_pdf_basis_diagonal(x, self.means, self.sigma_i)
        return nn.Dense(features=1, use_bias=False)(x)#.squeeze()

def get_initial_params(module, input_dims, key):
    init_shape = jnp.ones(input_dims, jnp.float32)
    initial_params = module.init(key, init_shape)['params']
    return initial_params

