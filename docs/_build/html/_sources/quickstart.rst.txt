Quickstart
==========

This quickstart guide will help you get up and running with JAX-bandflux for supernova light curve modeling. By the end of this guide, you'll be able to load supernova data, fit SALT3 parameters, and visualize the results.

Before You Begin
--------------

Make sure you have JAX-bandflux installed:

.. code-block:: bash

   pip install jax-bandflux

For GPU acceleration, ensure you have installed JAX with CUDA support (see :doc:`installation` for details).

Basic Example
------------

Here's a simple example that demonstrates how to use JAX-bandflux to fit SALT3 parameters to supernova light curve data:

.. code-block:: bash

   # Install JAX-bandflux
   pip install jax-bandflux
   
   # Download and run the example script
   wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/fmin_bfgs.py
   python fmin_bfgs.py

This example uses the L-BFGS-B optimization algorithm to fit SALT3 parameters to supernova light curve data. The script will:

1. Load supernova data for SN 19dwz
2. Register necessary bandpasses
3. Define an objective function based on chi-squared
4. Optimize the SALT3 parameters using L-BFGS-B
5. Print the best-fit parameters

The output should look something like:

.. code-block:: text

   Optimization successful: True
   Number of function evaluations: 87
   
   Best-fit parameters:
            z = 0.152300 (fixed)
           t0 = 58651.234567
           x0 = 1.234567e-05
           x1 = 0.987654
            c = 0.123456
   
   Final chi-squared: 123.45

Loading Data
-----------

JAX-bandflux provides flexible routines for loading supernova light curve data:

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

The returned values are:

* ``times``: JAX array of observation times (MJD)
* ``fluxes``: JAX array of flux measurements
* ``fluxerrs``: JAX array of flux measurement errors
* ``zps``: JAX array of zero points
* ``band_indices``: JAX array of indices mapping to registered bandpasses
* ``bridges``: Tuple of precomputed bridge data for efficient flux calculations
* ``fixed_z``: Tuple of (z, z_err) if ``fix_z=True``, else None

You can print information about the loaded data:

.. code-block:: python

   import jax.numpy as jnp
   
   print(f"Number of observations: {len(times)}")
   print(f"Unique bands: {len(jnp.unique(band_indices))}")
   if fixed_z is not None:
       print(f"Redshift: {fixed_z[0]:.4f} ± {fixed_z[1]:.4f}")

For lower-level access to the raw data, you can use the ``load_hsf_data`` function:

.. code-block:: python

   from jax_supernovae.data import load_hsf_data
   
   # Load raw data for a specific supernova
   data = load_hsf_data('19dwz', base_dir='data')

Fitting SALT Parameters
---------------------

JAX-bandflux provides efficient functions for calculating model fluxes and fitting SALT3 parameters to supernova light curve data. The SALT3 model has the following parameters:

* ``z``: Redshift
* ``t0``: Time of peak brightness (MJD)
* ``x0``: Amplitude parameter
* ``x1``: Stretch parameter (related to light curve width)
* ``c``: Color parameter

Here's a basic example of how to define an objective function for fitting SALT parameters:

.. code-block:: python

   import jax.numpy as jnp
   from jax_supernovae.salt3 import salt3_bandflux
   
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
       
       return chi2
You can then pass this objective function to your optimizer of choice, such as ``scipy.optimize.minimize`` or a nested sampling algorithm.

Here's an example using ``scipy.optimize.minimize``:

.. code-block:: python

   from scipy.optimize import minimize
   import numpy as np
   
   # Initial parameter values
   initial_params = np.array([0.1, 58650.0, 1e-5, 0.0, 0.0])  # z, t0, x0, x1, c
   
   # Parameter bounds
   bounds = [
       (0.01, 0.3),         # z
       (58500.0, 58700.0),  # t0
       (1e-6, 1e-4),        # x0
       (-3.0, 3.0),         # x1
       (-0.3, 0.3)          # c
   ]
   
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
   print("Best-fit parameters:", result.x)
   print("Final chi-squared:", result.fun)

Visualizing Results
-----------------

After fitting the SALT3 parameters, you can visualize the results using matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt
   from jax_supernovae.salt3 import salt3_bandflux
   
   # Create a dictionary with the best-fit parameters
   best_params = {
       'z': result.x[0],
       't0': result.x[1],
       'x0': result.x[2],
       'x1': result.x[3],
       'c': result.x[4]
   }
   
   # Compute model fluxes for all observations
   model_fluxes = []
   for i, (band_name, t, zp, zpsys) in enumerate(zip(data['band'], data['time'], data['zp'], data['zpsys'])):
       flux = salt3_bandflux(t, band_dict[band_name], best_params, zp=zp, zpsys=zpsys)
       model_fluxes.append(float(flux.ravel()[0]))
   
   model_fluxes = jnp.array(model_fluxes)
   
   # Plot the observed and model light curves
   plt.figure(figsize=(10, 6))
   
   # Get unique bands
   unique_bands = np.unique(data['band'])
   colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bands)))
   
   for i, band in enumerate(unique_bands):
       mask = data['band'] == band
       plt.errorbar(
           data['time'][mask] - best_params['t0'],
           data['flux'][mask],
           yerr=data['fluxerr'][mask],
           fmt='o',
           color=colors[i],
           label=f'{band} (Observed)'
       )
       plt.plot(
           data['time'][mask] - best_params['t0'],
           model_fluxes[mask],
           '-',
           color=colors[i],
           label=f'{band} (Model)'
       )
   
   plt.xlabel('Phase (days)')
   plt.ylabel('Flux')
   plt.title('Observed and Model Light Curves')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()
You can then pass this objective function to your optimizer of choice, such as ``scipy.optimize.minimize`` or a nested sampling algorithm.

Complete End-to-End Example
-------------------------

Here's a complete end-to-end example that you can run in under 5 minutes:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.optimize import minimize
   from jax_supernovae.data import load_and_process_data
   from jax_supernovae.salt3 import optimized_salt3_multiband_flux
   
   # Enable float64 precision for better accuracy
   jax.config.update("jax_enable_x64", True)
   
   # Load and process data
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',
       data_dir='data',
       fix_z=True
   )
   
   print(f"Loaded {len(times)} observations across {len(jnp.unique(band_indices))} bands")
   print(f"Redshift: {fixed_z[0]:.4f} ± {fixed_z[1]:.4f}")
   
   # Define the objective function
   def objective(parameters):
       # Create parameter dictionary
       param_dict = {
           'z': fixed_z[0],  # Fixed redshift
           't0': parameters[0],
           'x0': parameters[1],
           'x1': parameters[2],
           'c': parameters[3]
       }
       
       # Calculate model fluxes
       model_fluxes = optimized_salt3_multiband_flux(
           times, bridges, param_dict, zps=zps, zpsys='ab'
       )
       model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
       
       # Calculate chi-squared
       chi2 = float(jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2))
       
       return chi2
   
   # Initial parameter values
   initial_params = np.array([
       58650.0,    # t0
       1.5e-5,     # x0
       0.0,        # x1
       0.0         # c
   ])
   
   # Parameter bounds
   bounds = [
       (58500.0, 58700.0),  # t0
       (1e-6, 1e-4),        # x0
       (-3.0, 3.0),         # x1
       (-0.3, 0.3)          # c
   ]
   
   # Optimize the parameters
   result = minimize(
       objective,
       initial_params,
       method='L-BFGS-B',
       bounds=bounds,
       options={'disp': True}
   )
   
   # Print the results
   print("\nOptimization successful:", result.success)
   
   # Extract best-fit parameters
   best_params = {
       'z': fixed_z[0],
       't0': result.x[0],
       'x0': result.x[1],
       'x1': result.x[2],
       'c': result.x[3]
   }
   
   print("\nBest-fit parameters:")
   for name, value in best_params.items():
       print(f"{name:>10} = {value:.6f}")
   
   print(f"\nFinal chi-squared: {result.fun:.2f}")
   
   # Calculate model fluxes with best-fit parameters
   model_fluxes = optimized_salt3_multiband_flux(
       times, bridges, best_params, zps=zps, zpsys='ab'
   )
   model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
   
   # Plot the results
   plt.figure(figsize=(10, 6))
   
   # Get unique bands
   unique_bands = np.unique(band_indices)
   colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bands)))
   
   for i, band_idx in enumerate(unique_bands):
       mask = band_indices == band_idx
       plt.errorbar(
           times[mask] - best_params['t0'],
           fluxes[mask],
           yerr=fluxerrs[mask],
           fmt='o',
           color=colors[i],
           label=f'Band {band_idx} (Observed)'
       )
       plt.plot(
           times[mask] - best_params['t0'],
           model_fluxes[mask],
           '-',
           color=colors[i],
           label=f'Band {band_idx} (Model)'
       )
   
   plt.xlabel('Phase (days)')
   plt.ylabel('Flux')
   plt.title('Observed and Model Light Curves')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

Next Steps
---------

Now that you've seen the basics of JAX-bandflux, you can explore more advanced topics:

* :doc:`tutorials/basic_fitting` - A more detailed tutorial on fitting SALT3 parameters
* :doc:`tutorials/custom_bandpasses` - Learn how to use custom bandpasses
* :doc:`tutorials/nested_sampling` - Use nested sampling for Bayesian inference
* :doc:`guides/salt3_model` - In-depth guide to the SALT3 model
* :doc:`guides/bandpass_management` - Detailed guide to bandpass management
* :doc:`guides/jax_optimization` - Learn about JAX optimization techniques
* :doc:`api/index` - API reference documentation
* :doc:`examples` - Example scripts and notebooks