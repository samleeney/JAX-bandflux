# JAX Bandflux for Supernovae

A JAX-based implementation of supernova light curve modeling and analysis tools. This codebase provides efficient, differentiable implementations of core SNCosmo functionality using JAX.

## Installation

```bash
```bash
# Clone the repository, create and activate a virtual environment, and install dependencies (recommended)
git clone https://github.com/samleeney/jax-supernovae.git && cd JAX-bandflux && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

```bash
# For Nested Sampling, install blackjax and distrax
pip install git+https://github.com/handley-lab/blackjax@nested_sampling git+https://github.com/google-deepmind/distrax
```

## Usage
### Running the code
Users familiar with the `Using a custom fitter` [feature in sncosmo](https://sncosmo.readthedocs.io/en/stable/examples/plot_custom_fitter.html), will notice that this code follows a similar structure.

Simply define an objective function,

```python
def objective(parameters):
    # Create parameter dictionary
    params = {
        'z': parameters[0],
        't0': parameters[1],
        'x0': parameters[2],
        'x1': parameters[3],
        'c': parameters[4]
    }
    
    # Calculate model fluxes for all observations
    model_flux = []
    for i, (band_name, t, zp, zpsys) in enumerate(zip(data['band'], data['time'], 
                                                     data['zp'], data['zpsys'])):
        flux = salt3_bandflux(t, band_dict[band_name], params, 
                               zp=zp, zpsys=zpsys)
        # Extract the scalar value from the array
        flux_val = float(flux.ravel()[0])
        model_flux.append(flux_val)
        
    # Convert to array and calculate chi-squared
    model_flux = jnp.array(model_flux)
    chi2 = jnp.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)
    
    # Print overall chi-squared for debugging
    print(f"\nTotal chi-squared: {float(chi2):.2f}\n")
    
    return chi2

```

and pass it to your sampler of choice.

See the full example, which is analogous to the sncosmo example, in [fmin_bfgs.py](fmin_bfgs.py). We have also included a nested sampling implementation in [ns.py](ns.py).

Then run the code with:

```bash
python fmin_bfgs.py
```

### Data Loading

The codebase provides flexible data loading functionality for supernova light curve data, particularly optimised for HSF DR1 format data. There are several ways to load and process your data:

```python
from jax_supernovae.data import load_hsf_data

# Load data for a specific supernova
data = load_hsf_data('19agl', base_dir='data')
```

The loaded data will be an Astropy Table containing:
- `time`: Observation times (MJD)
- `band`: Filter/band names
- `flux`: Flux measurements
- `fluxerr`: Flux measurement errors
- `zp`: Zero points (defaults to 27.5 if not provided)

For analysis-ready JAX arrays and registered bandpasses:
```python
from jax_supernovae.data import load_and_process_data

# Load and process data with automatic bandpass registration
times, fluxes, fluxerrs, zps, band_indices, bridges = load_and_process_data(
    sn_name='19agl',
    data_dir='data'  # Optional, defaults to 'data'
)
```

This function:
1. Loads the raw data from the specified directory (defaults to 'data')
2. Registers all necessary bandpasses
3. Converts data to JAX arrays
4. Creates band indices for efficient processing
5. Precomputes bridge data (required for JAX optimisation) for each band

## Testing

This is a JAX implementation of the sncosmo bandflux function. The implementation is close to identical, but with a small change due to lack of availablity in JAX of a specific interpolation function. This leads to an ~0.001% different in bandflux results. Tests have been designed to confirm that after any changes, the match for key functions in the sncosmo version matches our implementation in JAX. If any changes are made to this section of the code these tests (which are automatically run as a github workflow) should be run to check that the matching is preserved. 

## Configuration Settings

The application's parameters are configured through a `settings.yaml` file in the root directory. This file contains all the hyperparameters for the nested sampling algorithm and prior distributions.

