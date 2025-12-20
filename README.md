# JAX-bandflux: Supernova SALT3 Model Fitting

[![PyPI version](https://badge.fury.io/py/jax-bandflux.svg)](https://badge.fury.io/py/jax-bandflux)
[![Tests](https://github.com/samleeney/JAX-bandflux/workflows/Tests/badge.svg)](https://github.com/samleeney/JAX-bandflux/actions)
[![Docs](https://img.shields.io/badge/docs-readthedocs-brightgreen)](https://jax-bandflux.readthedocs.io/)

**Author:** Samuel Alan Kossoff Leeney
**Homepage:** https://github.com/samleeney/JAX-bandflux
**Documentation:** https://jax-bandflux.readthedocs.io/

JAX-bandflux presents an implementation of supernova light curve modelling using JAX. The codebase offers a differentiable approach to core [SNCosmo](https://sncosmo.readthedocs.io/en/stable/) functionality implemented in JAX.

## Installation

### From PyPI (CPU by default)

```bash
pip install jax-bandflux
pip install --upgrade "jax[cpu]"
```

### GPU/CUDA wheels

Install the matching CUDA JAX wheel, e.g. for CUDA 12:

```bash
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax-bandflux
```

or with the extra marker:

```bash
pip install "jax-bandflux[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

We do not force a CUDA dependency in `install_requires`; see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other CUDA versions and matching wheels.

### Install from GitHub

```bash
pip install git+https://github.com/samleeney/JAX-bandflux.git
```

### Nested sampling extras

Optional dependencies for the nested sampling examples:

```bash
pip install "jax-bandflux[nested]"
```

### Development install

```bash
git clone https://github.com/samleeney/JAX-bandflux.git
cd JAX-bandflux
pip install -e ".[dev,nested,docs]"
```

> **Notes:**
> - Python >= 3.10. Core deps include JAX >= 0.4.20, NumPy >= 1.24.0, Astropy, and SNCosmo; SALT3/SALT3-NIR model files are bundled with the package (no GitHub install needed).
> - GPU support requires installing the appropriate `jax[cuda*]` wheel from the JAX release index. See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other CUDA versions.

## Quickstart

Run example analogous to [SNCosmo](https://sncosmo.readthedocs.io/en/stable/)'s "Using a custom fitter" example:

```bash
# Install from GitHub (recommended - contains latest features)
pip install git+https://github.com/samleeney/JAX-bandflux.git

# Download and run example
wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/fmin_bfgs.py
python fmin_bfgs.py
```

> **Note:** The latest features (including `SALT3Source` and `TimeSeriesSource`) are available on GitHub but not yet published to PyPI. For CUDA/GPU support, see the installation section below.

## Data format

Real light-curve data are simple ASCII tables per supernova (e.g., `data/<SN>/all.phot`) with required columns `time`/`mjd`, `band`/`bandpass`, `flux`, and `fluxerr`; `zp` defaults to 27.5 if omitted. A minimal template lives at `jax_supernovae/data/example_template.phot`. See the [data loading guide](docs/data_loading.rst) for column details, accepted band names, and mag→flux conversion tips.

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

> **Batched parameter evaluations:** When JAX-bandflux is used inside GPU-based samplers and parameters are evaluated in batches, the fused bandflux kernels deliver roughly two orders of magnitude speedup per parameter set compared to SNCosmo while matching fluxes to 0.001% (see Leeney et al. 2025).

See the [documentation](https://jax-bandflux.readthedocs.io/) for details.

## Testing

This repository implements the JAX version of the [SNCosmo](https://sncosmo.readthedocs.io/en/stable/) bandflux function. Although the implementations are nearly identical, a minor difference exists due to the absence of a specific interpolation function in JAX, resulting in a discrepancy of approximately 0.001% in bandflux results. Tests have been provided to confirm that key functions in the [SNCosmo](https://sncosmo.readthedocs.io/en/stable/) version correspond with our JAX implementation. It is recommended to run these tests, especially following any modifications.

```bash
pytest tests/ -v
```

## Contributing & Support

- See `CONTRIBUTING.md` for how to report issues and submit PRs.
- For help, open a GitHub issue with a minimal example and your environment (Python/JAX/JAXlib, CPU vs GPU, CUDA version).

## Academic Use

If you use JAX-bandflux in your research, please cite:

```bibtex
@article{leeney2025jax,
  title={JAX-bandflux: differentiable supernovae SALT modelling for cosmological analysis on GPUs},
  author={Leeney, Samuel Alan Kossoff},
  journal={arXiv preprint arXiv:2504.08081},
  year={2025}
}
```

## What is the `.airules` file?

The `.airules` file provides essential context to help language models understand and work with this codebase—particularly for new code that may not be included in model training datasets. It contains detailed information on:

- Data structures
- Core functions
- Implementation constraints
- Testing requirements

If you are using Cursor, rename this file to `.cursorrules` to enable automatic context integration.

## Contributing and support

- See `CONTRIBUTING.md` for how to report bugs, propose features, and open PRs.
- For questions/support, please open a GitHub issue with environment details (Python/JAX version), install path (PyPI/GitHub), and a minimal reproducer.
