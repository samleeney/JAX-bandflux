Quickstart
==========

This section demonstrates a minimal working example of jax_supernovae:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import jax_supernovae as js

   # Enable float64 precision
   jax.config.update("jax_enable_x64", True)

   # Load data for SN 19dwz
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = js.data.load_and_process_data('19dwz')
   # See :doc:`data_loading` for more details on data loading

   # Define SALT3 parameters
   params = {'z': fixed_z[0], 't0': 58650.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

   # Calculate model fluxes
   model_fluxes = js.salt3.salt3_bandflux(times, bridges, params, zp=zps)

   # Print chi-squared
   chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
   print(f"Chi-squared: {chi2:.2f}")

Expected output:

.. code-block:: text

   Chi-squared: 123.45