import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import sncosmo

# Load the model and data
model = sncosmo.Model(source='salt3')
data = sncosmo.load_example_data()

# Define the objective function using JAX
@jax.jit
def objective(parameters):
    # Set model parameters
    model.parameters = parameters

    # Evaluate model fluxes
    model_flux = model.bandflux(data['band'], data['time'],
                                zp=data['zp'], zpsys=data['zpsys'])

    # Calculate chi-squared using JAX numpy
    chi2 = jnp.sum(((data['flux'] - model_flux) / data['fluxerr']) ** 2)
    return chi2

# Initial parameter values (z, t0, x0, x1, c)
start_parameters = jnp.array([0.4, 55098., 1e-5, 0., 0.])

# Parameter bounds
bounds = [(0.3, 0.7), (55080., 55120.), (None, None), (None, None), (None, None)]

# Minimize the objective function
result = minimize(objective, start_parameters, method='L-BFGS-B', bounds=bounds)

# Extract and print optimized parameters
optimized_parameters = result.x
print(optimized_parameters)

# Test to compare outputs
import numpy as np

# Original parameters from sncosmo-fitter.py
original_parameters = np.array([0.49999872, 55099.458, 1.007e-05, 0.866, -0.057])

# Parameters from the JAX version
new_parameters = np.array(optimized_parameters)

# Compare the two sets of parameters
if np.allclose(original_parameters, new_parameters, atol=1e-6):
    print("Test passed: Outputs are identical within tolerance.")
else:
    print("Test failed: Outputs differ.") 