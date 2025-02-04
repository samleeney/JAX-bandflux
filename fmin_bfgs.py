import jax
import jax.numpy as jnp
from scipy.optimize import fmin_l_bfgs_b
import sncosmo
from jax_supernovae.bandpasses import Bandpass
from jax_supernovae.salt3nir import salt3nir_bandflux

# Enable float64 precision for better accuracy
jax.config.update("jax_enable_x64", True)

# Load example data from sncosmo
data = sncosmo.load_example_data()

# Convert sncosmo bandpasses to our JAX-compatible format
band_dict = {}
for band_name in set(data['band']):
    snc_band = sncosmo.get_bandpass(band_name)
    band_dict[band_name] = Bandpass(snc_band.wave, snc_band.trans)

# Define an objective function that we will pass to the minimiser
def objective(parameters):
    # Create parameter dictionary
    # Note: parameters[2] is now log10(x0)
    params = {
        'z': parameters[0],
        't0': parameters[1],
        'x0': 10**parameters[2],  # Convert from log space
        'x1': parameters[3],
        'c': parameters[4]
    }
    
    # Calculate model fluxes for all observations
    model_flux = []
    try:
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

# Starting parameter values [z, t0, log10(x0), x1, c]
# Note: x0 is now in log space
start_parameters = [0.4, 55098., -5., 0., 0.]  # Changed x0 to log10(x0)

# Parameter bounds
bounds = [(0.3, 0.7),      # z
         (55080., 55120.), # t0
         (-10., 0.),       # log10(x0) bounds
         (-3., 3.),        # x1
         (-0.3, 0.3)]      # c

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
param_names = ['z', 't0', 'log10(x0)', 'x1', 'c']
print("\nBest-fit parameters:")
for name, value in zip(param_names, parameters):
    print(f"{name:>10} = {value:.6f}")
print(f"\nFinal chi-squared: {val:.2f}") 