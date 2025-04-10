Generating Model Fluxes
=====================

This section explains how to calculate model fluxes using the SALT3 model in jax_supernovae.

SALT3 Parameters
-------------

The SALT3 model has the following parameters:

- ``z``: Redshift of the supernova
- ``t0``: Time of peak brightness (MJD)
- ``x0``: Amplitude parameter (overall flux normalization)
- ``x1``: Stretch parameter (related to light curve width)
- ``c``: Color parameter (related to supernova reddening)

These parameters are typically stored in a dictionary:

.. code-block:: python

   params = {
       'z': 0.1,      # Redshift
       't0': 58650.0, # Time of peak brightness
       'x0': 1e-5,    # Amplitude parameter
       'x1': 0.0,     # Stretch parameter
       'c': 0.0       # Color parameter
   }

Calculating Model Fluxes
---------------------

The ``salt3_bandflux`` function calculates model fluxes for given observation times and bandpasses:

.. code-block:: python

   from jax_supernovae.salt3 import salt3_bandflux

   # Define SALT3 parameters
   params = {
       'z': 0.1,      # Redshift
       't0': 58650.0, # Time of peak brightness
       'x0': 1e-5,    # Amplitude parameter
       'x1': 0.0,     # Stretch parameter
       'c': 0.0       # Color parameter
   }

   # Calculate model fluxes for all observations
   model_fluxes = salt3_bandflux(times, bridges, params, zp=zps)

   # Calculate chi-squared
   import jax.numpy as jnp
   chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
   print(f"Chi-squared: {chi2:.2f}")

Complete Example
-------------

This example demonstrates loading supernova data and calculating model fluxes:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax_supernovae.data import load_and_process_data
   from jax_supernovae.salt3 import salt3_bandflux

   # Enable float64 precision
   jax.config.update("jax_enable_x64", True)

   # Load data
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',
       data_dir='data',
       fix_z=True
   )

   # Define SALT3 parameters
   params = {
       'z': fixed_z[0],  # Use fixed redshift
       't0': 58650.0,    # Time of peak brightness
       'x0': 1e-5,       # Amplitude parameter
       'x1': 0.0,        # Stretch parameter
       'c': 0.0          # Color parameter
   }

   # Calculate model fluxes
   model_fluxes = salt3_bandflux(times, bridges, params, zp=zps)

   # Calculate chi-squared
   chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
   print(f"Chi-squared: {chi2:.2f}")