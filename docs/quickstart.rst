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
   times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = js.data.load_and_process_data('19dwz')
   # See :doc:`data_loading` for more details on data loading

   # Define SALT3 parameters
   params = {'z': fixed_z[0], 't0': 58650.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
   
   # To add dust extinction, include these parameters:
   # params.update({
   #     'dust_type': 0,  # 0=CCM89, 1=OD94, 2=F99
   #     'ebv': 0.1,      # E(B-V) reddening
   #     'r_v': 3.1       # R_V parameter (default: 3.1)
   # })

   # Calculate model fluxes
   model_fluxes = js.salt3.optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
   # Index the model fluxes with band_indices to match observations
   model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]

   # Print chi-squared
   chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
   print(f"Chi-squared: {chi2:.2f}")

Expected output:

.. code-block:: text

   Chi-squared: 123.45

Using Dust Extinction
--------------------

To apply dust extinction to the SALT3 model, add the dust parameters to the ``params`` dictionary:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import jax_supernovae as js
   
   # Enable float64 precision
   jax.config.update("jax_enable_x64", True)
   
   # Load data for SN 19dwz
   times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = js.data.load_and_process_data('19dwz')
   
   # Define SALT3 parameters with dust extinction
   params = {
       'z': fixed_z[0],
       't0': 58650.0,
       'x0': 1e-5,
       'x1': 0.0,
       'c': 0.0,
       'dust_type': 0,  # 0=CCM89, 1=OD94, 2=F99
       'ebv': 0.1,      # E(B-V) reddening
       'r_v': 3.1       # R_V parameter
   }
   
   # Calculate model fluxes with dust extinction
   model_fluxes = js.salt3.optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
   model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
   
   # Print chi-squared
   chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
   print(f"Chi-squared with dust: {chi2:.2f}")

For more details on the dust extinction implementation, see :doc:`dust_extinction_architecture` and :doc:`model_fluxes`.
