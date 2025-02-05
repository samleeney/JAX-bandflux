# JAX Bandflux for Supernovae

[![PyPI version](https://badge.fury.io/py/jax-bandflux.svg)](https://badge.fury.io/py/jax-bandflux)

This repository presents an implementation of supernova light curve modelling using JAX. The codebase offers a differentiable approach to core SNCosmo functionality implemented in JAX.

## Installation

To install the repository, please execute the following command:

```bash
git clone git@github.com:samleeney/JAX-bandflux.git && cd JAX-bandflux && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

For nested sampling functionality, install the required packages with:

```bash
pip install git+https://github.com/handley-lab/blackjax@nested_sampling distrax
```

## Usage

### Running the Code

This repository follows a structure similar to the `Using a custom fitter` example provided by SNCosmo. You may define the objective function as illustrated below:

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

To execute the code, run:

```bash
python fmin_bfgs.py
```

### Data Loading

The repository offers flexible routines for loading supernova light curve data, particularly optimised for HSF DR1 format. There are various methods to load and process your data.

To load data for a specific supernova:

```python
from jax_supernovae.data import load_hsf_data

# Load data for a specific supernova
data = load_hsf_data('19agl', base_dir='data')
```

The data is returned as an Astropy Table that includes:
- `time`: Observation times (MJD)
- `band`: Filter or band names
- `flux`: Flux measurements
- `fluxerr`: Errors associated with flux measurements
- `zp`: Zero points (defaults to 27.5 when not provided)

For analysis-ready JAX arrays and automatic bandpass registration, use:

```python
from jax_supernovae.data import load_and_process_data

# Load and process data with automatic bandpass registration
times, fluxes, fluxerrs, zps, band_indices, bridges = load_and_process_data(
    sn_name='19agl',
    data_dir='data'  # Optional, the default is 'data'
)
```

This function performs the following steps:
1. Loads raw data from the specified directory (default: 'data').
2. Registers the required bandpasses.
3. Converts data into JAX arrays.
4. Generates band indices for efficient processing.
5. Precomputes bridge data for each band, required for JAX optimisation.

## Testing

This repository implements the JAX version of the SNCosmo bandflux function. Although the implementations are nearly identical, a minor difference exists due to the absence of a specific interpolation function in JAX, resulting in a discrepancy of approximately 0.001% in bandflux results. Tests have been provided to confirm that key functions in the SNCosmo version correspond with our JAX implementation. It is recommended to run these tests, especially following any modifications.

## What is the .airules file?

Large Language Models are frequently used to optimise research and development. The `.airules` file provides context to help LLMs understand and work with this codebase. This is particularly important for new code that will not have been included in model training datasets. The file contains detailed information about data structures, core functions, critical implementation rules, and testing requirements. If you are using Cursor, rename this file to `.cursorrules` and it will be automatically interpreted as context.