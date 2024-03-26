from jax import numpy as jnp
from jax.scipy import stats as jstats

from jax_utils import Array, f32


# Gaussian normalization terms
def gaussian_norm_term(d: int, cov: Array) -> float:
    return jnp.sqrt((2 * jnp.pi)**d * jnp.linalg.det(cov))

def gaussian_norm_term_diagonal(d: int, sigma_i: Array) -> float:
    return jnp.sqrt((2 * jnp.pi)**d * jnp.prod(sigma_i))

def gaussian_norm_term_scalar(d: int, sigma: Array) -> float:
    return jnp.sqrt((2 * jnp.pi * sigma)**d)


# Gaussian functions
def mvn_pdf_diagonal(x, means, sigma_i):
    d = x.shape[0]
    log_mvn_pdf = - 0.5 * jnp.sum((x - means)**2 / sigma_i)
    norm_term = gaussian_norm_term_diagonal(d, sigma_i)
    return jnp.exp(log_mvn_pdf) / norm_term

# Gaussian basis functions
def mvn_pdf_basis_diagonal(x, means, sigma_i):

    d = x.shape[0]
    m = means.shape[0]

    log_mvn_pdf_basis = - 0.5 * jnp.sum(
        (x.reshape(1, d) - means.reshape(m, d))**2,
        axis=1,
    ) / sigma_i

    norm_term = gaussian_norm_term_diagonal(d, sigma_i)
    #return jnp.exp(log_mvn_pdf_basis) / norm_term
    return (jnp.exp(log_mvn_pdf_basis) / norm_term).squeeze()
