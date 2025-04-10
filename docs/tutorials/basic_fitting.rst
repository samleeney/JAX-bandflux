Basic Fitting Tutorial
====================

This tutorial demonstrates how to use JAX-bandflux to fit SALT3 parameters to supernova light curve data.

Prerequisites
-----------

Before starting this tutorial, make sure you have installed JAX-bandflux:

.. code-block:: bash

   pip install jax-bandflux

Loading Data
-----------

First, let's load some supernova light curve data:

.. code-block:: python

   from jax_supernovae.data import load_and_process_data
   
   # Load and process data with automatic bandpass registration
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',  # Name of the supernova
       data_dir='data',  # Optional, the default is 'data'
       fix_z=True        # Whether to load and fix redshift from redshifts.dat
   )

This function performs several steps:

1. Loads raw data from the specified directory
2. Registers all required bandpasses automatically
3. Converts data into JAX arrays for efficient computation
4. Generates band indices for optimized processing
5. Precomputes bridge data for each band
6. Optionally loads redshift data if ``fix_z=True``

For more details on data loading, including how to generate synthetic data, see :doc:`../data_loading`.

Defining the Objective Function
-----------------------------

Next, let's define an objective function for fitting SALT3 parameters:

.. code-block:: python

   import jax.numpy as jnp
   from jax_supernovae.salt3 import salt3_bandflux
   
   def objective(parameters):
       # Create a dictionary containing parameters
       params = {
           'z': fixed_z[0] if fixed_z is not None else parameters[0],
           't0': parameters[0 if fixed_z is not None else 1],
           'x0': parameters[1 if fixed_z is not None else 2],
           'x1': parameters[2 if fixed_z is not None else 3],
           'c': parameters[3 if fixed_z is not None else 4]
       }
       
       # Compute model fluxes for all observations
       model_fluxes = salt3_bandflux(times, bridges, params, zp=zps)
       
       # Calculate the chi-squared statistic
       chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
       
       return chi2

Optimizing the Parameters
-----------------------

Now, let's use the L-BFGS-B optimization algorithm to find the best-fit SALT3 parameters:

.. code-block:: python

   from scipy.optimize import minimize
   import numpy as np
   
   # Initial parameter values
   if fixed_z is not None:
       initial_params = np.array([0.0, 1e-5, 0.0, 0.0])  # t0, x0, x1, c
   else:
       initial_params = np.array([0.1, 0.0, 1e-5, 0.0, 0.0])  # z, t0, x0, x1, c
   
   # Parameter bounds
   if fixed_z is not None:
       bounds = [(-10, 10), (1e-10, 1e-2), (-3, 3), (-1, 1)]  # t0, x0, x1, c
   else:
       bounds = [(0.01, 0.2), (-10, 10), (1e-10, 1e-2), (-3, 3), (-1, 1)]  # z, t0, x0, x1, c
   
   # Optimize the parameters
   result = minimize(
       objective,
       initial_params,
       method='L-BFGS-B',
       bounds=bounds,
       options={'disp': True}
   )
   
   # Print the results
   print("Optimization successful:", result.success)
   print("Number of function evaluations:", result.nfev)
   
   # Extract the best-fit parameters
   if fixed_z is not None:
       best_t0, best_x0, best_x1, best_c = result.x
       best_z = fixed_z[0]
   else:
       best_z, best_t0, best_x0, best_x1, best_c = result.x
   
   print("Best-fit parameters:")
   print(f"z = {best_z:.6f}")
   print(f"t0 = {best_t0:.6f}")
   print(f"x0 = {best_x0:.6e}")
   print(f"x1 = {best_x1:.6f}")
   print(f"c = {best_c:.6f}")

Plotting the Results
-----------------

Finally, let's plot the observed and model light curves:

.. code-block:: python

   import matplotlib.pyplot as plt
   from jax_supernovae.salt3 import salt3_bandflux
   
   # Create a dictionary with the best-fit parameters
   best_params = {
       'z': best_z,
       't0': best_t0,
       'x0': best_x0,
       'x1': best_x1,
       'c': best_c
   }
   
   # Compute model fluxes for all observations
   model_fluxes = salt3_bandflux(times, bridges, best_params, zp=zps)
   
   # Create a figure
   plt.figure(figsize=(10, 6))
   
   # Plot the observed and model light curves
   unique_bands = np.unique(band_indices)
   colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bands)))
   
   for i, band_idx in enumerate(unique_bands):
       mask = band_indices == band_idx
       plt.errorbar(
           times[mask] - best_t0,
           fluxes[mask],
           yerr=fluxerrs[mask],
           fmt='o',
           color=colors[i],
           label=f'Band {band_idx}'
       )
       plt.plot(
           times[mask] - best_t0,
           model_fluxes[mask],
           '-',
           color=colors[i]
       )
   
   plt.xlabel('Phase (days)')
   plt.ylabel('Flux')
   plt.legend()
   plt.title('Observed and Model Light Curves')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

Complete Example
--------------

Here's the complete example:

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.optimize import minimize
   from jax_supernovae.data import load_and_process_data
   from jax_supernovae.salt3 import salt3_bandflux
   
   # Load and process data
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',
       data_dir='data',
       fix_z=True
   )
   
   # Define the objective function
   def objective(parameters):
       # Create a dictionary containing parameters
       params = {
           'z': fixed_z[0] if fixed_z is not None else parameters[0],
           't0': parameters[0 if fixed_z is not None else 1],
           'x0': parameters[1 if fixed_z is not None else 2],
           'x1': parameters[2 if fixed_z is not None else 3],
           'c': parameters[3 if fixed_z is not None else 4]
       }
       
       # Compute model fluxes for all observations
       model_fluxes = salt3_bandflux(times, bridges, params, zp=zps)
       
       # Calculate the chi-squared statistic
       chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
       
       return chi2
   
   # Initial parameter values
   if fixed_z is not None:
       initial_params = np.array([0.0, 1e-5, 0.0, 0.0])  # t0, x0, x1, c
   else:
       initial_params = np.array([0.1, 0.0, 1e-5, 0.0, 0.0])  # z, t0, x0, x1, c
   
   # Parameter bounds
   if fixed_z is not None:
       bounds = [(-10, 10), (1e-10, 1e-2), (-3, 3), (-1, 1)]  # t0, x0, x1, c
   else:
       bounds = [(0.01, 0.2), (-10, 10), (1e-10, 1e-2), (-3, 3), (-1, 1)]  # z, t0, x0, x1, c
   
   # Optimize the parameters
   result = minimize(
       objective,
       initial_params,
       method='L-BFGS-B',
       bounds=bounds,
       options={'disp': True}
   )
   
   # Print the results
   print("Optimization successful:", result.success)
   print("Number of function evaluations:", result.nfev)
   
   # Extract the best-fit parameters
   if fixed_z is not None:
       best_t0, best_x0, best_x1, best_c = result.x
       best_z = fixed_z[0]
   else:
       best_z, best_t0, best_x0, best_x1, best_c = result.x
   
   print("Best-fit parameters:")
   print(f"z = {best_z:.6f}")
   print(f"t0 = {best_t0:.6f}")
   print(f"x0 = {best_x0:.6e}")
   print(f"x1 = {best_x1:.6f}")
   print(f"c = {best_c:.6f}")
   
   # Compute model fluxes for all observations
   best_params = {
       'z': best_z,
       't0': best_t0,
       'x0': best_x0,
       'x1': best_x1,
       'c': best_c
   }
   model_fluxes = salt3_bandflux(times, bridges, best_params, zp=zps)
   
   # Create a figure
   plt.figure(figsize=(10, 6))
   
   # Plot the observed and model light curves
   unique_bands = np.unique(band_indices)
   colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bands)))
   
   for i, band_idx in enumerate(unique_bands):
       mask = band_indices == band_idx
       plt.errorbar(
           times[mask] - best_t0,
           fluxes[mask],
           yerr=fluxerrs[mask],
           fmt='o',
           color=colors[i],
           label=f'Band {band_idx}'
       )
       plt.plot(
           times[mask] - best_t0,
           model_fluxes[mask],
           '-',
           color=colors[i]
       )
   
   plt.xlabel('Phase (days)')
   plt.ylabel('Flux')
   plt.legend()
   plt.title('Observed and Model Light Curves')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()