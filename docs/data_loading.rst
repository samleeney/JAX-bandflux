Data Loading
============

JAX-bandflux provides utilities for loading supernova photometry data and
preparing it for model fitting.

Synthetic Data
--------------

For testing and development, you can create synthetic observations:

.. doctest::

   >>> # Create synthetic observation times
   >>> times = jnp.array([58650.0, 58655.0, 58660.0, 58665.0, 58670.0])
   >>>
   >>> # Synthetic fluxes and errors
   >>> fluxes = jnp.array([100.0, 150.0, 180.0, 160.0, 120.0])
   >>> fluxerrs = jnp.array([5.0, 6.0, 7.0, 6.5, 5.5])
   >>>
   >>> # All observations in one band
   >>> unique_bands = ['bessellb']
   >>> band_indices = jnp.zeros(5, dtype=jnp.int32)
   >>>
   >>> print(f"Created {len(times)} observations")
   Created 5 observations
   >>> print(f"Flux range: {float(jnp.min(fluxes)):.1f} to {float(jnp.max(fluxes)):.1f}")
   Flux range: 100.0 to 180.0

Multi-Band Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate observations across multiple bands:

.. doctest::

   >>> # Generate multi-band synthetic data
   >>> source = SALT3Source()
   >>> params = {'x0': 1e-4, 'x1': 0.5, 'c': 0.0}
   >>> z = 0.05
   >>>
   >>> # Observation setup
   >>> obs_times = np.array([0, 5, 10, 15, 20])  # Days from peak
   >>> bands = ['bessellb', 'bessellv', 'bessellr']
   >>>
   >>> # Generate observations
   >>> np.random.seed(42)
   >>> all_times, all_fluxes, all_errors, all_bands = [], [], [], []
   >>>
   >>> for band in bands:
   ...     phases = obs_times / (1 + z)
   ...     true_flux = np.array(source.bandflux(params, band, phases, zp=27.5, zpsys='ab'))
   ...     noise = np.random.normal(0, np.abs(true_flux) * 0.05)
   ...     all_times.extend(obs_times)
   ...     all_fluxes.extend(true_flux + noise)
   ...     all_errors.extend(np.abs(true_flux) * 0.05)
   ...     all_bands.extend([band] * len(obs_times))
   >>>
   >>> print(f"Generated {len(all_times)} observations across {len(bands)} bands")
   Generated 15 observations across 3 bands
   >>> print(f"Band distribution: {[(b, all_bands.count(b)) for b in bands]}")
   Band distribution: [('bessellb', 5), ('bessellv', 5), ('bessellr', 5)]

Data Structure
--------------

For fitting, you need the following arrays:

.. doctest::

   >>> # Convert to JAX arrays
   >>> times = jnp.array(all_times)
   >>> fluxes = jnp.array(all_fluxes)
   >>> fluxerrs = jnp.array(all_errors)
   >>> zps = jnp.full(len(times), 27.5)
   >>>
   >>> # Band indexing for optimized mode
   >>> unique_bands = ['bessellb', 'bessellv', 'bessellr']
   >>> band_to_idx = {b: i for i, b in enumerate(unique_bands)}
   >>> band_indices = jnp.array([band_to_idx[b] for b in all_bands])
   >>>
   >>> # Pre-compute bridges
   >>> bridges = tuple(precompute_bandflux_bridge(get_bandpass(b))
   ...                 for b in unique_bands)
   >>> print(f"Pre-computed {len(bridges)} bridges for bands: {unique_bands}")
   Pre-computed 3 bridges for bands: ['bessellb', 'bessellv', 'bessellr']
   >>>
   >>> # Inspect bridge structure
   >>> print(f"B-band bridge wavelength grid: {bridges[0]['wave'].shape[0]} points")
   B-band bridge wavelength grid: 400 points
   >>> print(f"Grid spacing: {bridges[0]['dwave']} Angstroms")
   Grid spacing: 5.0 Angstroms

Required Data Arrays
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Array
     - Type
     - Description
   * - ``times``
     - float array
     - Observation times (MJD or days from reference)
   * - ``fluxes``
     - float array
     - Observed flux values
   * - ``fluxerrs``
     - float array
     - Flux uncertainties (1-sigma)
   * - ``zps``
     - float array
     - Zero points for each observation (typically 27.5 for AB mags)
   * - ``band_indices``
     - int array
     - Index into ``unique_bands`` for each observation
   * - ``bridges``
     - tuple
     - Pre-computed bandpass integration grids
   * - ``unique_bands``
     - list
     - List of unique bandpass names

Loading Real Data
-----------------

For real supernova data in HSF format, use ``load_and_process_data``:

.. code-block:: python

   from jax_supernovae.data import load_and_process_data

   # Load data for a specific supernova
   result = load_and_process_data('19dwz', fix_z=True)
   times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = result

   # fixed_z contains (z, z_err) if fix_z=True
   z, z_err = fixed_z
   print(f"Redshift: {z:.4f} ± {z_err:.4f}")

This function:

1. Loads photometry from ``data/photometry/{sn_name}.dat``
2. Loads redshift from ``data/redshifts.dat`` (with fallback to ``data/targets.dat``)
3. Registers all required bandpasses
4. Pre-computes bridges for each unique band
5. Returns all arrays ready for fitting

Data File Format
~~~~~~~~~~~~~~~~

The HSF photometry format expects tab-separated columns:

.. code-block:: text

   # time band flux fluxerr zp
   58650.0  bessellb  123.45  6.17  27.5
   58651.0  bessellv  156.78  7.84  27.5
   ...

Redshifts
~~~~~~~~~

Redshifts are loaded from two possible sources:

1. **Primary**: ``data/redshifts.dat`` - High-quality spectroscopic redshifts
2. **Fallback**: ``data/targets.dat`` - All targets with potentially lower-quality redshifts

If you set ``fix_z=True``, the loader looks in ``redshifts.dat``. To provide your
own value, add a line with:

``SN instrument z_hel plus minus flag``

For example: ``19dwz SNIFS 0.04608 5.2e-06 7.8e-07 s``

Using the Data
--------------

Once loaded, compute model fluxes:

.. doctest::

   >>> # Data summary
   >>> print(f"Data: {len(times)} observations, {len(set(all_bands))} bands")
   Data: 15 observations, 3 bands
   >>>
   >>> # Compute model fluxes
   >>> z = 0.05
   >>> t0 = 0.0
   >>> phases = (times - t0) / (1 + z)
   >>>
   >>> model = source.bandflux(
   ...     params, None, phases, zp=zps, zpsys='ab',
   ...     band_indices=band_indices,
   ...     bridges=bridges,
   ...     unique_bands=unique_bands
   ... )
   >>> print(f"Computed {len(model)} model fluxes")
   Computed 15 model fluxes
   >>>
   >>> # Compare observed vs model
   >>> print("First 5 observations:")
   First 5 observations:
   >>> for i in range(5):
   ...     print(f"  Time {float(times[i]):7.1f}: obs={float(fluxes[i]):7.2f} ± {float(fluxerrs[i]):.2f}, model={float(model[i]):.2f}")
     Time     0.0: obs= 639.50 ± 31.20, model=624.01
     Time     5.0: obs= 534.18 ± 26.90, model=537.90
     Time    10.0: obs= 415.89 ± 20.14, model=402.84
     Time    15.0: obs= 278.35 ± 12.93, model=258.65
     Time    20.0: obs= 159.07 ± 8.05, model=160.96

Computing Chi-Squared
~~~~~~~~~~~~~~~~~~~~~

.. doctest::

   >>> chi2 = jnp.sum(((fluxes - model) / fluxerrs)**2)
   >>> print(f"Chi-squared: {float(chi2):.2f} for {len(fluxes)} data points")
   Chi-squared: 13.84 for 15 data points
   >>> print(f"Reduced chi-squared: {float(chi2) / (len(fluxes) - 3):.2f}")
   Reduced chi-squared: 1.15

Preparing Your Own Data
-----------------------

The loader expects a simple ASCII table per supernova. By default it looks for
``data/<SN_NAME>/all.phot`` (or any ``.phot``/``.dat`` containing the object name).

**Required columns (case-insensitive aliases in parentheses):**

- ``time`` (or ``mjd``): observation times in MJD
- ``band`` (or ``bandpass``): filter name matching a registered band
- ``flux``: calibrated flux in linear units consistent with ``zp``/``zpsys``
- ``fluxerr``: 1-sigma uncertainty on ``flux``

**Optional columns:**

- ``zp``: zero point (defaults to 27.5 if missing)
- ``zpsys``: zero-point system, typically ``ab``

**Band names recognised by default:** ``g, r, i, z, ztfg, ztfr, c, o, H`` plus
``bessellb, bessellv, bessellr, besselli, bessellux`` (from sncosmo). Custom
bandpasses can be registered via ``register_all_bandpasses(custom_bandpass_files=...)``.

Converting Magnitudes to Flux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your data are in magnitudes, convert to flux:

.. doctest::

   >>> zp = 23.9
   >>> mag = 20.0
   >>> magerr = 0.05
   >>>
   >>> flux_conv = 10 ** (-0.4 * (mag - zp))
   >>> fluxerr_conv = flux_conv * np.log(10) * 0.4 * magerr
   >>> print(f"Magnitude {mag:.1f} mag → Flux {flux_conv:.2f} ± {fluxerr_conv:.2f}")
   Magnitude 20.0 mag → Flux 36.31 ± 1.67
   >>>
   >>> # Example: converting multiple magnitudes
   >>> mags = np.array([19.0, 20.0, 21.0, 22.0])
   >>> fluxes_from_mag = 10 ** (-0.4 * (mags - zp))
   >>> print("Magnitude → Flux conversion:")
   Magnitude → Flux conversion:
   >>> for m, f in zip(mags, fluxes_from_mag):
   ...     print(f"  {m:.1f} mag → {f:.2f}")
     19.0 mag → 91.20
     20.0 mag → 36.31
     21.0 mag → 14.45
     22.0 mag → 5.75

Multiple Supernovae
-------------------

For fitting multiple supernovae simultaneously:

.. code-block:: python

   from jax_supernovae.data import load_multiple_supernovae

   # Load multiple supernovae with shared band structure
   sn_names = ['19dwz', '19agl', '19bcf']
   data = load_multiple_supernovae(sn_names, fix_z=True)

   # Access data for all supernovae
   print(f"Loaded {data['n_sne']} supernovae")
   print(f"Unique bands: {data['unique_bands']}")

   # Individual supernova data
   for i, name in enumerate(sn_names):
       times_i = data['times_list'][i]
       print(f"{name}: {len(times_i)} observations")

   # Combined data for joint fitting
   all_times = data['all_times']
   sn_indices = data['sn_indices']  # Which SN each observation belongs to

See :doc:`sampling` for examples of joint fitting with nested sampling.
