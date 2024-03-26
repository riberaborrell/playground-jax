# jax

## Contains
Collection of notebooks about:
* **Autodiff:** jax automatic differentiation for real functions, real-valued functions (scalar) and vector-valued functions with and without batch input.
* **Regression problem:**: fitting problem for real-valued functions 1d and 2d with diferent basic neural network architectures with equinox and flax.
* **Sampling:** simulating brownian motion (toy sde) with finite time horizon and first hitting times
* **Stochastic optimization problems:** the loss function has to be sampled at each iteration. Finite time horizon and first hitting time horizon.


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
