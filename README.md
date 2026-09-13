# JAX Playground

## Contains
Collection of notebooks about:

#### Autodifferentiation
* Jax automatic differentiation for real functions, real-valued functions (scalar), and vector-valued functions with and without batch input.
* Jacovian-vector product (JVP) and vector-Jacobin products (VJP)

#### Regression Problem
* Simple linear regression
* One-dimensional regression problem (simple SGD)
* Regression problem for real-valued functions 1d and 2d with different type of optimizers with `optax` and basic neural network architectures with `equinox` and `flax`.

#### Sampling
* simulation of a _Brownian motion_ (toy sde) with finite time horizon and first hitting time


## Install

1) clone the repo
```
git clone git@github.com:riberaborrell/playground-jax.git
```

2) move inside the directory, create virtual environment (venv) and install required packages
```
cd playground-jax
make venv
```

3) activate venv
```
source venv/bin/activate
```


## Usage
```
cd notebooks
jupyter lab
```
