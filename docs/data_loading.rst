Data Loading
===========

This section provides a concise guide to loading and generating data for use with jax_supernovae.

Generating Synthetic Data
------------------------

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np

   # Generate synthetic observation times (in MJD)
   times = jnp.linspace(58640, 58680, 20)

   # Generate synthetic fluxes with noise
   true_fluxes = 1e-5 * jnp.exp(-((times - 58650)**2) / 100)
   flux_errors = true_fluxes * 0.1  # 10% errors
   observed_fluxes = true_fluxes + flux_errors * np.random.normal(size=len(times))
   observed_fluxes = jnp.array(observed_fluxes)
   flux_errors = jnp.array(flux_errors)

   # Generate band indices (assuming 2 bands)
   band_indices = jnp.array([0 if i % 2 == 0 else 1 for i in range(len(times))])

   # Generate zero points
   zps = jnp.ones_like(times) * 27.5

Loading Real Data
---------------

.. code-block:: python

   from jax_supernovae.data import load_and_process_data

   # Load data for a specific supernova
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',  # Name of the supernova
       data_dir='data',  # Directory containing the data
       fix_z=True        # Whether to fix the redshift
   )

   print(f"Loaded {len(times)} observations across {len(jnp.unique(band_indices))} bands")
   print(f"Redshift: {fixed_z[0]:.4f} ± {fixed_z[1]:.4f}")

Data Structure
------------

The data used in jax_supernovae consists of the following components:

- **times**: Observation times in Modified Julian Date (MJD)
- **fluxes**: Observed flux values
- **fluxerrs**: Flux measurement errors
- **zps**: Zero points for flux calibration
- **band_indices**: Indices mapping observations to bandpasses
- **bridges**: Precomputed data for efficient flux calculations
- **fixed_z**: Tuple of (redshift, redshift_error) if fix_z=True