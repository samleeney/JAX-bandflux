# JAX Bandflux for Supernovae SALT3 model fitting

[![PyPI version](https://badge.fury.io/py/jax-bandflux.svg)](https://badge.fury.io/py/jax-bandflux)

This repository presents an implementation of supernova light curve modelling using JAX. The codebase offers a differentiable approach to core SNCosmo functionality implemented in JAX.

## Quickstart
Run example analagous to SNCosmo's `Using a custom fitter` example:

```bash
pip install jax-bandflux
wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/fmin_bfgs.py
python fmin_bfgs.py
```

## Usage

### Data Loading

The repository offers flexible routines for loading supernova light curve data, particularly optimised for HSF DR1 format. The primary method for loading data is through the `load_and_process_data` function:

```python
from jax_supernovae.data import load_and_process_data

# Load and process data with automatic bandpass registration
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
    sn_name='19dwz',  # Name of the supernova
    data_dir='data',  # Optional, the default is 'data'
    fix_z=True        # Whether to load and fix redshift from redshifts.dat
)
```

This function performs several steps:
1. Loads raw data from the specified directory
2. Registers all required bandpasses automatically
3. Converts data into JAX arrays for efficient computation
4. Generates band indices for optimised processing
5. Precomputes bridge data for each band
6. Optionally loads redshift data if `fix_z=True`

The returned values are:
- `times`: JAX array of observation times (MJD)
- `fluxes`: JAX array of flux measurements
- `fluxerrs`: JAX array of flux measurement errors
- `zps`: JAX array of zero points
- `band_indices`: JAX array of indices mapping to registered bandpasses
- `bridges`: Tuple of precomputed bridge data for efficient flux calculations
- `fixed_z`: Tuple of (z, z_err) if `fix_z=True`, else None

For lower-level access to the raw data, you can use the `load_hsf_data` function:

```python
from jax_supernovae.data import load_hsf_data

# Load raw data for a specific supernova
data = load_hsf_data('19dwz', base_dir='data')
```

The data is returned as an Astropy Table that includes:
- `time`: Observation times (MJD)
- `band`: Filter or band names
- `flux`: Flux measurements
- `fluxerr`: Errors associated with flux measurements
- `zp`: Zero points (defaults to 27.5 when not provided)

### Custom Model Files

The package includes SALT3 model files in the `sncosmo-modelfiles/models` directory. Three model variants are available:

- `salt3-nir`: Extended SALT3 model with near-infrared coverage (2800-17000Å)
- `salt3`: Standard SALT3 model (2800-12000Å)

Each model directory contains the following key files:
- `salt3_template_0.dat`: M0 component (mean SN Ia spectrum)
- `salt3_template_1.dat`: M1 component (spectral variation)
- `salt3_color_correction.dat`: Colour law coefficients
- `SALT3.INFO`: Model metadata and configuration
- Additional files for variance and covariance

To use a custom model, ensure your model files follow this structure and place them in a subdirectory of `sncosmo-modelfiles/models`. The model files should contain:

```plaintext
# salt3_template_[0/1].dat format:
phase wavelength value
...

# salt3_color_correction.dat format:
ncoeff
coeff1
coeff2
...
coeffn
Salt2ExtinctionLaw.version 1
Salt2ExtinctionLaw.min_lambda value
Salt2ExtinctionLaw.max_lambda value
```

The package will automatically handle model file loading and interpolation in a JAX-compatible way.

### Fitting SALT parameters

```python
def objective(parameters):
    # Create a dictionary containing parameters
    params = {
        'z': parameters[0],
        't0': parameters[1],
        'x0': parameters[2],
        'x1': parameters[3],
        'c': parameters[4]
    }
    
    # Compute model fluxes for all observations
    model_flux = []
    for i, (band_name, t, zp, zpsys) in enumerate(zip(data['band'], data['time'], data['zp'], data['zpsys'])):
        flux = salt3_bandflux(t, band_dict[band_name], params, zp=zp, zpsys=zpsys)
        # Extract the scalar value from the array
        flux_val = float(flux.ravel()[0])
        model_flux.append(flux_val)
    
    # Convert to a JAX array and calculate the chi-squared statistic
    model_flux = jnp.array(model_flux)
    chi2 = jnp.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)
    
    # Display the total chi-squared for debugging purposes
    print(f"\nTotal chi-squared: {float(chi2):.2f}\n")
    
    return chi2
```

Pass this function to your sampler of choice. A complete example, analogous to the SNCosmo case, is provided in [fmin_bfgs.py](fmin_bfgs.py). A nested sampling implementation is also available in [ns.py](ns.py).

## Testing

This repository implements the JAX version of the SNCosmo bandflux function. Although the implementations are nearly identical, a minor difference exists due to the absence of a specific interpolation function in JAX, resulting in a discrepancy of approximately 0.001% in bandflux results. Tests have been provided to confirm that key functions in the SNCosmo version correspond with our JAX implementation. It is recommended to run these tests, especially following any modifications.

## What is the .airules file?

Large Language Models are frequently used to optimise research and development. The `.airules` file provides context to help LLMs understand and work with this codebase. This is particularly important for new code that will not have been included in model training datasets. The file contains detailed information about data structures, core functions, critical implementation rules, and testing requirements. If you are using Cursor, rename this file to `.cursorrules` and it will be automatically interpreted as context.