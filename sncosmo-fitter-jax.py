import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import basinhopping, fmin_l_bfgs_b
import sncosmo
from jax_supernovae.models import Model
from jax_supernovae.salt2 import salt2_flux
from jax_supernovae.salt2_data import get_salt2_wave_grid

# Initialize JAX model
jax_model = Model()
# Get wavelength grid from our data module
jax_model.wave = get_salt2_wave_grid()
jax_model.flux = lambda t, w: salt2_flux(t, w, jax_model.parameters)

# Load the data
data = sncosmo.load_example_data()

# Pre-compute JAX arrays for data we'll use repeatedly
jax_data_flux = jnp.array(data['flux'])
jax_fluxerr = jnp.array(data['fluxerr'])

def get_model_flux(parameters):
    """Helper function to get model flux using numpy arrays"""
    param_dict = {
        'z': parameters[0],
        't0': parameters[1],
        'x0': parameters[2],
        'x1': parameters[3],
        'c': parameters[4]
    }
    jax_model.parameters = param_dict
    return jax_model.bandflux(data['band'], data['time'],
                         zp=data['zp'], zpsys=data['zpsys'])

@jax.jit
def compute_chi2(model_flux, data_flux, fluxerr):
    """Pure JAX function to compute chi-squared"""
    return jnp.sum(((data_flux - model_flux) / fluxerr) ** 2)

def objective(parameters):
    """Main objective function"""
    # Get model flux (using numpy)
    model_flux = get_model_flux(parameters)
    
    # Convert to JAX array and compute chi-squared
    jax_model_flux = jnp.array(model_flux)
    return float(compute_chi2(jax_model_flux, jax_data_flux, jax_fluxerr))

# Initial parameter values and bounds
start_parameters = np.array([0.4, 55098., 1e-5, 0., 0.])
bounds = [(0.3, 0.7), (55080., 55120.), (None, None), (None, None),
          (None, None)]

# Define a minimizer for use with basinhopping
def minimizer_wrapper(func, x0, **kwargs):
    result = fmin_l_bfgs_b(func, x0, bounds=bounds, approx_grad=True)
    return result[0], result[1]

# Run basin-hopping to find global minimum
result = basinhopping(
    objective,
    start_parameters,
    minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds},
    niter=20,  # Number of basin-hopping iterations
    T=1.0,     # Temperature parameter for acceptance
    stepsize=0.1  # Initial step size for perturbation
)

# Extract optimized parameters and print in consistent format
print("RESULT:", result.x.tolist()) 