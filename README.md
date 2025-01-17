# JAX Supernovae

A JAX implementation of supernova light curve models, based on SNCosmo.

## Features
- JAX-based implementation for GPU acceleration and automatic differentiation
- Compatible with SNCosmo models and bandpasses
- Efficient computation of bandpass-integrated fluxes
- Support for AB magnitude system

## Installation
```bash
pip install -e .
```

## Usage
```python
import jax.numpy as jnp
from jax_supernovae import Model, get_bandpass, get_magsystem

# Create a model
model = Model()

# Set parameters
model.parameters = {'t0': 55000.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0, 'z': 0.5}

# Calculate bandflux
times = jnp.linspace(54950, 55050, 100)
flux = model.bandflux('sdssg', times)
```

## Testing
```bash
python -m pytest
```

## License
MIT License 