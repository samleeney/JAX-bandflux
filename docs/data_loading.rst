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
   times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = load_and_process_data(
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
- **unique_bands**: List of band names corresponding to band_indices/bridges
- **bridges**: Precomputed data for efficient flux calculations
- **fixed_z**: Tuple of (redshift, redshift_error) if fix_z=True

Preparing your own observations
--------------------------------

The loader expects a simple ASCII table per supernova. By default it looks for
``data/<SN_NAME>/all.phot`` (or any ``.phot``/``.dat`` containing the object
name). You can point to another root with ``data_dir=...``.

**Required columns (case-insensitive aliases in parentheses):**

- ``time`` (or ``mjd``): observation times in MJD
- ``band`` (or ``bandpass``): filter name matching a registered band
- ``flux``: calibrated flux in linear units consistent with ``zp``/``zpsys``
- ``fluxerr``: 1-sigma uncertainty on ``flux``

**Optional columns:**

- ``zp``: zero point (defaults to 27.5 if missing)
- ``zpsys``: zero-point system, typically ``ab`` (currently informational)
- any extra columns (e.g., ``mag``, ``magerr``) are ignored by the loader

**Band names recognised by default:** ``g, r, i, z, ztfg, ztfr, c, o, H`` plus
``bessellb, bessellv, bessellr, besselli, bessellux`` (from sncosmo). Custom
bandpasses can be registered via ``register_all_bandpasses(custom_bandpass_files=...)``.

**Minimal file template:** ``jax_supernovae/data/example_template.phot`` shows
the expected header and ordering. You can copy it next to your data and replace
the rows with your measurements.

**Converting magnitudes to flux:** if your data are in magnitudes, convert to
flux using the same zero point and system you plan to store:

.. code-block:: python

   import numpy as np

   zp = 23.9  # use the same value you write into the file
   flux = 10 ** (-0.4 * (mag - zp))
   fluxerr = flux * np.log(10) * 0.4 * magerr

Redshifts
---------

If you set ``fix_z=True``, the loader looks in ``redshifts.dat`` (packaged in
``jax_supernovae/data`` by default). To provide your own value, add a line with:

``SN instrument z_hel plus minus flag``

for example:

``19dwz SNIFS 0.04608 5.2e-06 7.8e-07 s``

You can also pass a custom ``redshift_file`` to ``load_redshift`` if you keep
redshifts elsewhere.
