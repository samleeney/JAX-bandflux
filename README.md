# JAX-bandflux: Supernova SALT3 Model Fitting

[![PyPI version](https://badge.fury.io/py/jax-bandflux.svg)](https://badge.fury.io/py/jax-bandflux)
[![Tests](https://github.com/samleeney/JAX-bandflux/workflows/Tests/badge.svg)](https://github.com/samleeney/JAX-bandflux/actions)

**Author:** Samuel Alan Kossoff Leeney
**Homepage:** https://github.com/samleeney/JAX-bandflux
**Documentation:** https://jax-bandflux.readthedocs.io/

JAX-bandflux presents an implementation of supernova light curve modelling using JAX. The codebase offers a differentiable approach to core [SNCosmo](https://sncosmo.readthedocs.io/en/stable/) functionality implemented in JAX.

## Installation

### Install from PyPI

```bash
pip install jax-bandflux
```

### Install from GitHub

```bash
pip install git+https://github.com/samleeney/JAX-bandflux.git
```

### For development

```bash
git clone https://github.com/samleeney/JAX-bandflux.git
cd JAX-bandflux
pip install -e .
```

## Dependencies

JAX-bandflux requires:

- Python >= 3.10
- JAX >= 0.4.20 (with CUDA support for GPU acceleration)
- NumPy >= 1.24.0
- SNCosmo >= 2.9.0
- BlackJAX (for nested sampling: requires Handley Lab fork, not yet merged with main branch)
- Distrax (for probability distributions)

> **Note:** For nested sampling examples, you must install the Handley Lab fork of BlackJAX (not yet merged with main branch):
> ```bash
> pip install git+https://github.com/handley-lab/blackjax@proposal
> ```
> See: https://handley-lab.co.uk/nested-sampling-book/intro.html

## Quickstart

Run example analogous to [SNCosmo](https://sncosmo.readthedocs.io/en/stable/)'s "Using a custom fitter" example:

```bash
pip install jax-bandflux
wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/fmin_bfgs.py
python fmin_bfgs.py
```

## API Compatibility with SNCosmo

JAX-bandflux provides an API similar to SNCosmo's SALT3Source, with key differences for JAX compatibility:

### Functional Parameter API

Parameters are passed as dictionaries to methods rather than stored as object attributes. This is a **hard constraint** for JIT compilation - JAX requires pure functional code where all inputs are explicit arguments.

**SNCosmo approach:**
```python
source.set(z=0.5, t0=0, x0=1e-5, x1=0.5, c=0.1)
flux = source.bandflux('bessellb', time=10)
```

**JAX-bandflux approach:**
```python
from jax_supernovae import SALT3Source

source = SALT3Source()
params = {'x0': 1e-5, 'x1': 0.5, 'c': 0.1}
flux = source.bandflux(params, 'bessellb', phase=10/(1+0.5))
```

This enables JIT compilation while maintaining numerical accuracy within **0.001%** of SNCosmo.

### Performance Optimization with Bridges

The `bridges` parameter allows precomputed filter integration grids, providing **~100x speedup** for repeated calculations (e.g., nested sampling):

```python
from jax_supernovae.salt3 import precompute_bandflux_bridge

# Precompute once
bridges = [precompute_bandflux_bridge(bp) for bp in bandpasses]

# Reuse thousands of times in JIT-compiled functions
@jax.jit
def likelihood(params):
    flux = source.bandflux(params, None, phases,
                          bridges=bridges,
                          band_indices=indices,
                          unique_bands=bands)
    return -0.5 * chi2
```

**What are bridges?** Precomputed wavelength grids with interpolated filter transmission values. Instead of interpolating the filter for every likelihood evaluation, you compute it once and reuse it. For nested sampling with 10,000+ evaluations, this provides a massive speedup.

See the [documentation](https://jax-bandflux.readthedocs.io/) for details.

## Testing

This repository implements the JAX version of the [SNCosmo](https://sncosmo.readthedocs.io/en/stable/) bandflux function. Although the implementations are nearly identical, a minor difference exists due to the absence of a specific interpolation function in JAX, resulting in a discrepancy of approximately 0.001% in bandflux results. Tests have been provided to confirm that key functions in the [SNCosmo](https://sncosmo.readthedocs.io/en/stable/) version correspond with our JAX implementation. It is recommended to run these tests, especially following any modifications.

```bash
pytest tests/ -v
```

## What is the `.airules` file?

The `.airules` file provides essential context to help language models understand and work with this codebase—particularly for new code that may not be included in model training datasets. It contains detailed information on:

- Data structures
- Core functions
- Implementation constraints
- Testing requirements

If you are using Cursor, rename this file to `.cursorrules` to enable automatic context integration.
