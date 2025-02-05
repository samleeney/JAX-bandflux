import jax
import jax.numpy as jnp
from scipy.optimize import fmin_l_bfgs_b
import sncosmo
import yaml
from jax_supernovae.bandpasses import Bandpass
from jax_supernovae.salt3nir import salt3nir_bandflux
from jax_supernovae.data import load_redshift

# Enable float64 precision for better accuracy
jax.config.update("jax_enable_x64", True)

# Load settings
with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

# Load example data from sncosmo
data = sncosmo.load_example_data()

# Convert sncosmo bandpasses to our JAX-compatible format
band_dict = {}
for band_name in set(data['band']):
    snc_band = sncosmo.get_bandpass(band_name)
    band_dict[band_name] = Bandpass(snc_band.wave, snc_band.trans)

# Check if we should fix z
fix_z = settings.get('fix_z', False)
fixed_z = None
if fix_z:
    try:
        z, z_err, flag = load_redshift('19agl')  # Replace with actual SN name if needed
        fixed_z = z
        print(f"Using fixed redshift z = {z:.6f} ± {z_err:.6f} (flag: {flag})")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not load redshift: {e}")
        fix_z = False

# Define an objective function that we will pass to the minimiser
def objective(parameters):
    # Create parameter dictionary
    # Note: parameters[1] or [2] (depending on fix_z) is now log10(x0)
    try:
        if fix_z:
            params = {
                'z': fixed_z,
                't0': parameters[0],
                'x0': 10**parameters[1],  # Convert from log space
                'x1': parameters[2],
                'c': parameters[3]
            }
        else:
            params = {
                'z': parameters[0],
                't0': parameters[1],
                'x0': 10**parameters[2],  # Convert from log space
                'x1': parameters[3],
                'c': parameters[4]
            }
        
        # Calculate model fluxes for all observations
        model_flux = []
        for i, (band_name, t, zp, zpsys) in enumerate(zip(data['band'], data['time'], 
                                                         data['zp'], data['zpsys'])):
            flux = salt3nir_bandflux(t, band_dict[band_name], params, 
                                   zp=zp, zpsys=zpsys)
            # Extract the scalar value from the array
            flux_val = float(flux.ravel()[0])
            # Check for invalid values
            if not jnp.isfinite(flux_val):
                return 1e12  # Return a large but finite value
            model_flux.append(flux_val)
            
        # Convert to array and calculate chi-squared
        model_flux = jnp.array(model_flux)
        chi2 = jnp.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)
        
        # Guard against non-finite values
        if not jnp.isfinite(chi2):
            return 1e12
        
        # Print overall chi-squared for debugging
        print(f"\nTotal chi-squared: {float(chi2):.2f}\n")
        
        return float(chi2)  # Ensure we return a float
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e12  # Return a large but finite value

# Starting parameter values
if fix_z:
    start_parameters = [
        55098.,     # t0
        -5.,        # log10(x0)
        0.,         # x1
        0.          # c
    ]
    # Parameter bounds
    bounds = [
        (55080., 55120.),  # t0
        (-10., 0.),        # log10(x0)
        (-3., 3.),         # x1
        (-0.3, 0.3)        # c
    ]
else:
    start_parameters = [
        0.4,        # z
        55098.,     # t0
        -5.,        # log10(x0)
        0.,         # x1
        0.          # c
    ]
    # Parameter bounds
    bounds = [
        (0.3, 0.7),       # z
        (55080., 55120.), # t0
        (-10., 0.),       # log10(x0)
        (-3., 3.),        # x1
        (-0.3, 0.3)       # c
    ]

print("\nStarting optimisation...")
print("Initial parameters:", start_parameters)

# Run the optimisation with more conservative settings
parameters, val, info = fmin_l_bfgs_b(objective, start_parameters,
                                     bounds=bounds,
                                     approx_grad=True,
                                     maxfun=100,        # Limit number of function evaluations
                                     maxiter=15,        # Limit number of iterations
                                     factr=1e7,         # Increase convergence tolerance
                                     epsilon=1e-8)      # Smaller finite difference step

# Print results
if fix_z:
    param_names = ['t0', 'log10(x0)', 'x1', 'c']
else:
    param_names = ['z', 't0', 'log10(x0)', 'x1', 'c']

print("\nBest-fit parameters:")
for name, value in zip(param_names, parameters):
    print(f"{name:>10} = {value:.6f}")
if fix_z:
    print(f"{'z':>10} = {fixed_z:.6f} (fixed)")
print(f"\nFinal chi-squared: {val:.2f}") 