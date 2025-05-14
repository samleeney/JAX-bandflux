Generating Model Fluxes
=====================

This section explains how to calculate model fluxes using the SALT3 model in jax_supernovae.

SALT3 Parameters
-------------

The SALT3 model is parameterized by the following variables:

- ``z``: Redshift of the supernova
- ``t0``: Time of peak brightness (MJD)
- ``x0``: Amplitude parameter (overall flux normalization)
- ``x1``: Stretch parameter (related to light curve width)
- ``c``: Color parameter (related to supernova reddening)

Additionally, dust extinction can be applied using the following optional parameters:

- ``dust_type``: Integer index for the dust law (0=CCM89, 1=OD94, 2=F99)
- ``ebv``: E(B-V) reddening value
- ``r_v``: R_V parameter (default: 3.1)

These parameters are typically stored in a dictionary:

.. code-block:: python

   # Basic SALT3 parameters
   params = {
       'z': 0.1,      # Redshift
       't0': 58650.0, # Time of peak brightness
       'x0': 1e-5,    # Amplitude parameter
       'x1': 0.0,     # Stretch parameter
       'c': 0.0       # Color parameter
   }

   # With dust extinction parameters
   params_with_dust = {
       'z': 0.1,      # Redshift
       't0': 58650.0, # Time of peak brightness
       'x0': 1e-5,    # Amplitude parameter
       'x1': 0.0,     # Stretch parameter
       'c': 0.0,      # Color parameter
       'dust_type': 0, # 0=CCM89, 1=OD94, 2=F99
       'ebv': 0.1,     # E(B-V) reddening
       'r_v': 3.1      # R_V parameter
   }

Calculating Model Fluxes
---------------------

The ``optimized_salt3_multiband_flux`` function calculates model fluxes for given observation times and bandpasses:

.. code-block:: python

   from jax_supernovae.salt3 import optimized_salt3_multiband_flux

   # Calculate model fluxes for all observations
   model_fluxes = optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')

   # Index the model fluxes with band_indices
   model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]

   # Calculate chi-squared
   import jax.numpy as jnp
   chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)

Complete Example
-------------

This example demonstrates loading supernova data and calculating model fluxes:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax_supernovae.data import load_and_process_data
   from jax_supernovae.salt3 import optimized_salt3_multiband_flux

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

   # Calculate model fluxes for all bandpasses
   model_fluxes = optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')

   # Index the model fluxes with band_indices to match observations
   model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]

   # Calculate chi-squared
   chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
   print(f"Chi-squared: {chi2:.2f}")

The ``optimized_salt3_multiband_flux`` function returns a 2D array with shape (n_times, n_bands). To match these model fluxes with the observed fluxes, we use ``band_indices`` to select the appropriate band for each observation time.

Dust Extinction
--------------

JAX-bandflux supports three dust extinction laws:

1. CCM89: Cardelli, Clayton, Mathis (1989) extinction law
2. OD94: O'Donnell (1994) extinction law
3. F99: Fitzpatrick (1999) extinction law

To apply dust extinction to the SALT3 model, add the following parameters to the ``params`` dictionary:

.. code-block:: python

   # Add dust extinction parameters
   params['dust_type'] = 0  # 0=CCM89, 1=OD94, 2=F99
   params['ebv'] = 0.1      # E(B-V) reddening
   params['r_v'] = 3.1      # R_V parameter (default: 3.1)

The dust extinction is applied to the rest-frame spectral energy distribution (SED) before integration over the bandpass. This allows for accurate modeling of the effects of dust on the observed supernova light curves.

For more details on the dust extinction implementation, see :doc:`dust_extinction_architecture`.
