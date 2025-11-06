# TimeSeriesSource Documentation

## Overview

`TimeSeriesSource` is a JAX-bandflux class for fitting custom supernova spectral energy distributions (SEDs). It provides a JAX/GPU-accelerated implementation matching sncosmo's `TimeSeriesSource` API whilst using a functional parameter-passing approach for optimal performance in MCMC and nested sampling applications.

## Key Features

- **Custom SED Models**: Fit any spectral time series defined on a 2D (phase × wavelength) grid
- **Bicubic Interpolation**: Matches sncosmo exactly using JAX primitives
- **Functional API**: Parameters passed as dictionaries for JAX compatibility
- **Two-Tier Performance**: Simple mode for convenience, optimised mode for speed
- **JIT Compatible**: Works seamlessly in JIT-compiled likelihood functions
- **GPU Accelerated**: Runs efficiently on GPUs via JAX
- **Numerical Accuracy**: Matches sncosmo to <0.01% (tested)

## API Comparison: sncosmo vs JAX-bandflux

### Constructor (Nearly Identical)

**sncosmo:**
```python
source = sncosmo.TimeSeriesSource(phase, wave, flux,
                                   zero_before=False,
                                   time_spline_degree=3,
                                   name=None, version=None)
```

**JAX-bandflux:**
```python
source = TimeSeriesSource(phase, wave, flux,  # Same signature!
                          zero_before=False,
                          time_spline_degree=3,
                          name=None, version=None)
```

### Method Calls (Functional API)

**sncosmo (stateful):**
```python
source.set(amplitude=1.0)
flux = source.bandflux('bessellb', 0.0, zp=25.0, zpsys='ab')
```

**JAX-bandflux (functional):**
```python
params = {'amplitude': 1.0}
flux = source.bandflux(params, 'bessellb', 0.0, zp=25.0, zpsys='ab')
```

The key difference: JAX-bandflux passes parameters as a dictionary to each method call, enabling JAX to trace parameter dependencies for autodiff and JIT compilation.

## Basic Usage

### Creating a TimeSeriesSource

```python
import numpy as np
from jax_supernovae import TimeSeriesSource

# Define your model grid
phase = np.linspace(-20, 50, 100)  # Days
wave = np.linspace(3000, 9000, 200)  # Angstroms

# Create flux array (phase × wavelength)
# This could come from:
# - Theoretical models
# - Spectroscopic observations
# - PCA components
# - Machine learning models
p_grid, w_grid = np.meshgrid(phase, wave, indexing='ij')
flux = ...  # Your 2D flux array (erg/s/cm²/Å)

# Create source
source = TimeSeriesSource(phase, wave, flux,
                          zero_before=False,  # Extrapolate before minphase
                          time_spline_degree=3,  # Cubic interpolation
                          name='my_model')
```

### Simple Photometry

```python
# Define parameters
params = {'amplitude': 1.0}

# Single observation
flux_b = source.bandflux(params, 'bessellb', 0.0, zp=25.0, zpsys='ab')

# Light curve (multiple phases, same band)
phases = np.linspace(-10, 30, 50)
fluxes_b = source.bandflux(params, 'bessellb', phases, zp=25.0, zpsys='ab')

# Multi-band observation (same phase, different bands)
bands = ['bessellb', 'bessellv', 'bessellr']
phases_same = np.zeros(3)
fluxes_multi = source.bandflux(params, bands, phases_same, zp=25.0, zpsys='ab')

# Multi-band light curve (equal-length arrays)
bands_lc = ['bessellb', 'bessellv', 'bessellr', 'bessellb', 'bessellv']
phases_lc = np.array([0, 0, 0, 5, 5])
fluxes_lc = source.bandflux(params, bands_lc, phases_lc, zp=25.0, zpsys='ab')
```

### Calculate Magnitudes

```python
# Magnitude in AB system
mag_b = source.bandmag(params, 'bessellb', 'ab', 0.0)

# Multiple magnitudes
mags = source.bandmag(params, 'bessellb', 'ab', phases)
```

## High-Performance Mode

For MCMC, nested sampling, or any application requiring many model evaluations, use the optimised mode with pre-computed bridges:

```python
import jax.numpy as jnp
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge

# Example: 100 observations in 3 bands
n_obs = 100
phases = jnp.linspace(-10, 40, n_obs)
band_names = ['bessellb', 'bessellv', 'bessellr'] * (n_obs // 3)  # Cycle through bands
zps = jnp.ones(n_obs) * 25.0

# Pre-compute bridges ONCE (outside the likelihood)
unique_bands = ['bessellb', 'bessellv', 'bessellr']
bridges = tuple(precompute_bandflux_bridge(get_bandpass(b))
                for b in unique_bands)

# Create band indices mapping each observation to its bridge
band_to_idx = {'bessellb': 0, 'bessellv': 1, 'bessellr': 2}
band_indices = jnp.array([band_to_idx[b] for b in band_names])

# Fast calculation (10-100x faster than simple mode)
params = {'amplitude': 1.0}
fluxes = source.bandflux(params, None, phases,  # Note: bands=None in optimised mode
                         zp=zps, zpsys='ab',
                         band_indices=band_indices,
                         bridges=bridges,
                         unique_bands=unique_bands)
```

## JIT-Compiled Likelihood Functions

TimeSeriesSource works seamlessly in JIT-compiled functions:

```python
import jax

# Define JIT-compiled likelihood
@jax.jit
def loglikelihood(amplitude, observed_fluxes, flux_errors):
    """Calculate log-likelihood for given amplitude."""
    params = {'amplitude': amplitude}

    # Calculate model fluxes (uses optimised mode)
    model_fluxes = source.bandflux(params, None, phases,
                                   zp=zps, zpsys='ab',
                                   band_indices=band_indices,
                                   bridges=bridges,
                                   unique_bands=unique_bands)

    # Chi-squared
    chi2 = jnp.sum(((observed_fluxes - model_fluxes) / flux_errors)**2)

    return -0.5 * chi2

# Use in optimization or sampling
from jaxopt import ScipyMinimize
solver = ScipyMinimize(fun=lambda amp: -loglikelihood(amp, data_fluxes, data_errors))
result = solver.run(1.0)  # Initial guess: amplitude=1.0
```

## Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase` | array_like | Required | 1D array of phase values (days). Must be sorted ascending. |
| `wave` | array_like | Required | 1D array of wavelength values (Å). Must be sorted ascending. |
| `flux` | array_like | Required | 2D array of flux values (erg/s/cm²/Å). Shape: (len(phase), len(wave)). |
| `zero_before` | bool | False | If True, flux is zero for phase < minphase. If False, extrapolates. |
| `time_spline_degree` | int | 3 | Time interpolation degree: 1 (linear) or 3 (cubic). |
| `name` | str | None | Optional name for the model. |
| `version` | str | None | Optional version identifier. |

### Model Parameters (Functional API)

The functional API requires passing parameters as a dictionary to each method call:

| Parameter | Type | Description |
|-----------|------|-------------|
| `amplitude` | float | Scaling factor for the model flux. |

**Example:**
```python
params = {'amplitude': 1.0}
flux = source.bandflux(params, 'bessellb', 0.0)
```

## Methods

### `bandflux(params, bands, phases, zp=None, zpsys=None, **kwargs)`

Calculate bandflux through specified bandpass(es).

**Parameters:**
- `params` (dict): Must contain `'amplitude'`
- `bands` (str, list, or None): Bandpass name(s). Use None for optimised mode.
- `phases` (float or array): Rest-frame phase(s) in days
- `zp` (float or array, optional): Zero point(s)
- `zpsys` (str, optional): Zero point system ('ab', etc.)
- `band_indices` (array, optional): For optimised mode
- `bridges` (tuple, optional): For optimised mode
- `unique_bands` (list, optional): For optimised mode

**Returns:**
- float or array: Bandflux value(s) matching input shape

### `bandmag(params, bands, magsys, phases, **kwargs)`

Calculate magnitude through specified bandpass(es).

**Parameters:**
- `params` (dict): Must contain `'amplitude'`
- `bands` (str or list): Bandpass name(s)
- `magsys` (str): Magnitude system ('ab', etc.)
- `phases` (float or array): Rest-frame phase(s)
- Additional kwargs for optimised mode

**Returns:**
- float or array: Magnitude value(s). Returns NaN for flux ≤ 0.

### Properties

- `param_names`: List of parameter names (['amplitude'])
- `minphase()`: Minimum phase of model (days)
- `maxphase()`: Maximum phase of model (days)
- `minwave()`: Minimum wavelength of model (Å)
- `maxwave()`: Maximum wavelength of model (Å)

## Advanced Topics

### Interpolation Methods

TimeSeriesSource supports two interpolation methods:

**Cubic Interpolation (default):**
```python
source = TimeSeriesSource(phase, wave, flux, time_spline_degree=3)
```
- Uses bicubic interpolation (same as sncosmo)
- Smooth light curves
- Better for well-sampled grids

**Linear Interpolation:**
```python
source = TimeSeriesSource(phase, wave, flux, time_spline_degree=1)
```
- Uses bilinear interpolation
- Faster computation
- Better for coarse grids or performance-critical applications

### Zero-Before Behaviour

**`zero_before=False` (default):**
- Extrapolates flux for phases before `minphase`
- Uses edge values from the grid
- Suitable for models where early-time flux is uncertain

**`zero_before=True`:**
```python
source = TimeSeriesSource(phase, wave, flux, zero_before=True)
```
- Returns exactly zero for phase < `minphase`
- Suitable for models that should not have flux before explosion
- Matches sncosmo's behaviour

### Creating Models from Real Data

TimeSeriesSource can be created from various sources. See `examples/custom_sed_example.py` for complete working examples.

**From sncosmo Spectral Templates:**
```python
import sncosmo
import numpy as np

# Load Hsiao Type Ia template
hsiao = sncosmo.get_source('hsiao')

# Define grids
phases = np.arange(-20.0, 86.0, 1.0)
wavelengths = np.arange(2000.0, 10000.0, 10.0)

# Generate 2D flux grid
flux_grid = np.zeros((len(phases), len(wavelengths)))
for i, phase in enumerate(phases):
    flux_grid[i, :] = hsiao._flux(phase, wavelengths)

# Create TimeSeriesSource
source = TimeSeriesSource(phases, wavelengths, flux_grid,
                          zero_before=True, time_spline_degree=3,
                          name='hsiao', version='sncosmo')
```

**From Spectroscopic Observations (Files):**
```python
# Load phase values
phases = np.loadtxt('spectra/phases.txt')

# Load wavelength grid
wavelengths = np.loadtxt('spectra/wavelengths.txt')

# Load flux grid (one file per epoch)
flux_grid = []
for phase in phases:
    filename = f'spectra/spectrum_day{int(phase):+03d}.txt'
    flux = np.loadtxt(filename)
    flux_grid.append(flux)
flux_grid = np.array(flux_grid)

source = TimeSeriesSource(phases, wavelengths, flux_grid)
```

**From Analytical/Theoretical Models:**
```python
import numpy as np
from jax_supernovae import TimeSeriesSource

# Define grids
phases = np.linspace(-20, 40, 80)
wavelengths = np.linspace(2500, 9000, 500)

# Create mesh grids for vectorised calculation
phase_grid, wave_grid = np.meshgrid(phases, wavelengths, indexing='ij')

# Simple physical model: expanding photosphere
t0 = 17.0  # Characteristic timescale (days)
L0 = 1.0e43  # Peak luminosity (erg/s)
T0 = 10000  # Initial temperature (K)

# Time-dependent luminosity (rise and decline)
time_since_explosion = phase_grid
luminosity = L0 * (time_since_explosion / t0)**2 * np.exp(-time_since_explosion / t0)
luminosity = np.maximum(luminosity, 0)

# Time-dependent temperature (cooling)
temperature = T0 * (1 + time_since_explosion / 30.0)**(-0.5)

# Blackbody SED (simplified)
h = 6.626e-27  # Planck's constant (erg s)
c = 3.0e10  # Speed of light (cm/s)
k_B = 1.381e-16  # Boltzmann constant (erg/K)

wave_cm = wave_grid * 1e-8
B_lambda = (2 * h * c**2 / wave_cm**5) / (np.exp(h * c / (wave_cm * k_B * temperature)) - 1)

# Convert to flux (assume distance and solid angle)
distance = 10 * 3.086e18  # 10 pc in cm
radius = 1e14  # Photosphere radius (cm)
solid_angle = np.pi * radius**2 / distance**2
flux_grid = B_lambda * solid_angle * (luminosity / L0) * 1e8  # Convert /cm to /Å

# Create TimeSeriesSource
source = TimeSeriesSource(phases, wavelengths, flux_grid,
                          zero_before=True, time_spline_degree=3,
                          name='analytical_expanding_photosphere')
```

**Fitting to Photometric Data:**
```python
import jax
import jax.numpy as jnp
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge

# Observed photometry
obs_phases = jnp.array([-10, -5, 0, 5, 10, 15])
obs_bands = ['bessellb', 'bessellb', 'bessellv', 'bessellv', 'bessellr', 'bessellr']
obs_fluxes = jnp.array([...])  # Your observed fluxes
flux_errors = jnp.array([...])  # Your flux uncertainties

# Set up optimised mode
unique_bands = ['bessellb', 'bessellv', 'bessellr']
bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)
band_to_idx = {band: i for i, band in enumerate(unique_bands)}
band_indices = jnp.array([band_to_idx[band] for band in obs_bands])

# JIT-compiled likelihood function
@jax.jit
def neg_log_likelihood(amplitude):
    """Negative log-likelihood for fitting."""
    params = {'amplitude': amplitude}
    model_fluxes = source.bandflux(
        params, None, obs_phases,
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands
    )
    chi2 = jnp.sum(((obs_fluxes - model_fluxes) / flux_errors)**2)
    return 0.5 * chi2

# Fit (example: grid search)
test_amplitudes = jnp.linspace(0.5, 4.0, 100)
chi2_values = jnp.array([neg_log_likelihood(a) for a in test_amplitudes])
best_amplitude = test_amplitudes[jnp.argmin(chi2_values)]
```

**Complete Working Examples:** See `examples/custom_sed_example.py` (tested, runs successfully):
- Example 1: Loading Hsiao template from sncosmo (validates to 0.003% accuracy)
- Example 2: Creating SEDs from file formats
- Example 3: Analytical expanding photosphere model (full blackbody physics)
- Example 4: Fitting custom SEDs to photometric data (with JIT-compiled likelihood)
- Example 5: Comparing custom SEDs with SALT3

## Performance Tips

1. **Use Optimised Mode for Fitting**: Pre-compute bridges once, reuse many times
2. **JIT Compile Likelihood Functions**: Use `@jax.jit` for 10-100x speedup
3. **Batch Observations**: Process multiple observations together when possible
4. **Appropriate Grid Resolution**: Balance accuracy vs memory/compute
5. **Use GPU When Available**: JAX automatically uses GPU if available

## Comparison with SALT3Source

| Feature | TimeSeriesSource | SALT3Source |
|---------|------------------|-------------|
| Model Type | Custom SED | SALT3-NIR only |
| Parameters | amplitude | x0, x1, c |
| Flexibility | Any 2D flux grid | Fixed SALT3 model |
| Use Case | Custom models, rare events | Type Ia SNe standardisation |
| Performance | Comparable | Comparable |

Both classes coexist and can be used together in the same analysis.

## Testing

Comprehensive tests verify TimeSeriesSource matches sncosmo to <0.01%:

```bash
cd JAX-bandflux
python tests/test_timeseries_source.py
```

Tests cover:
- Interpolation accuracy (linear and cubic)
- Bandflux calculations
- Zero-before behaviour
- Simple vs optimised mode consistency
- JIT compilation
- Return types and error handling

## Examples

See `examples/timeseries_source_demo.py` for comprehensive demonstrations including:
- Creating simple models
- Calculating synthetic photometry
- High-performance mode
- JIT-compiled likelihoods
- Comparison with sncosmo
- Plotting light curves

## Common Issues

### Q: Why does my model return NaN?

**A:** Check that:
1. Your phase/wavelength ranges cover the requested observations
2. Flux values are finite (no NaN/Inf in input grid)
3. For magnitudes: flux must be positive

### Q: Why is simple mode slow?

**A:** Simple mode creates bandpass bridges on-the-fly. For repeated calculations (MCMC/nested sampling), use optimised mode with pre-computed bridges.

### Q: How do I handle redshift?

**A:** TimeSeriesSource works in rest-frame. Calculate rest-frame phases outside:
```python
z = 0.5
t0 = 58650.0
times_obs = ...  # Observer-frame times
phases_rest = (times_obs - t0) / (1 + z)

# Then use rest-frame phases
flux = source.bandflux(params, band, phases_rest)
```

### Q: Can I use this with nested sampling?

**A:** Yes! TimeSeriesSource is designed for this. Use optimised mode:
```python
from jaxns import NestedSampler

def likelihood(amplitude):
    params = {'amplitude': amplitude}
    model_fluxes = source.bandflux(params, None, phases,
                                   band_indices=bi, bridges=br,
                                   unique_bands=ub)
    return -0.5 * jnp.sum(((data - model_fluxes) / errors)**2)

ns = NestedSampler(likelihood, ...)
results = ns.run()
```

## References

- sncosmo documentation: https://sncosmo.readthedocs.io/
- JAX documentation: https://jax.readthedocs.io/
- JAX-bandflux repository: [link to repo]

## Contributing

Found a bug or have a feature request? Please open an issue on GitHub.

## License

JAX-bandflux is released under [LICENSE]. See LICENSE file for details.
