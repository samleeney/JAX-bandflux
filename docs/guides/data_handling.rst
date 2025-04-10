Data Handling
===========

This guide provides in-depth information about data handling in JAX-bandflux.

Introduction
-----------

JAX-bandflux provides flexible routines for loading and processing supernova light curve data. The package is particularly optimized for the HSF DR1 format, but can be adapted to work with other data formats as well.

Loading Data
----------

The primary method for loading data is through the ``load_and_process_data`` function:

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

- ``times``: JAX array of observation times (MJD)
- ``fluxes``: JAX array of flux measurements
- ``fluxerrs``: JAX array of flux measurement errors
- ``zps``: JAX array of zero points
- ``band_indices``: JAX array of indices mapping to registered bandpasses
- ``bridges``: Tuple of precomputed bridge data for efficient flux calculations
- ``fixed_z``: Tuple of (z, z_err) if ``fix_z=True``, else None

For lower-level access to the raw data, you can use the ``load_hsf_data`` function:

.. code-block:: python

   from jax_supernovae.data import load_hsf_data
   
   # Load raw data for a specific supernova
   data = load_hsf_data('19dwz', base_dir='data')

The data is returned as an Astropy Table that includes:

- ``time``: Observation times (MJD)
- ``band``: Filter or band names
- ``flux``: Flux measurements
- ``fluxerr``: Errors associated with flux measurements
- ``zp``: Zero points (defaults to 27.5 when not provided)

Data Directory Structure
---------------------

JAX-bandflux expects a specific directory structure for supernova data:

.. code-block::

   data/
   ├── redshifts.dat
   ├── 19dwz/
   │   ├── all.phot
   │   ├── J_1D3.phot
   │   ├── J_2D.phot
   │   ├── ebv_fits.dat
   │   ├── max_fits.dat
   │   └── salt_fits.dat
   └── 20aai/
       └── ...

The ``redshifts.dat`` file contains redshift information for all supernovae:

.. code-block::

   # name z z_err
   19dwz 0.1234 0.0012
   20aai 0.0567 0.0008
   ...

Each supernova has its own directory containing photometric data files:

- ``all.phot``: Combined photometric data from all filters
- ``*.phot``: Filter-specific photometric data
- ``*_fits.dat``: Fitting results from previous analyses

Photometric Data Format
--------------------

The photometric data files (``*.phot``) are in a simple text format:

.. code-block::

   # time band flux fluxerr zp zpsys
   58765.123 ztfg 1.234e-5 5.678e-6 27.5 ab
   58766.456 ztfr 2.345e-5 6.789e-6 27.5 ab
   ...

The columns are:

- ``time``: Observation time in Modified Julian Date (MJD)
- ``band``: Filter or band name
- ``flux``: Flux measurement
- ``fluxerr``: Flux measurement error
- ``zp``: Zero point (defaults to 27.5 if not provided)
- ``zpsys``: Magnitude system (defaults to 'ab' if not provided)

Loading Redshift Data
------------------

You can load redshift data using the ``load_redshift`` function:

.. code-block:: python

   from jax_supernovae.data import load_redshift
   
   # Load redshift data for a specific supernova
   z, z_err = load_redshift('19dwz', base_dir='data')
   
   print(f"Redshift: {z} ± {z_err}")

This function reads the redshift information from the ``redshifts.dat`` file.

Converting Data to JAX Arrays
--------------------------

JAX-bandflux converts data to JAX arrays for efficient computation. This is done automatically by the ``load_and_process_data`` function, but you can also do it manually:

.. code-block:: python

   import jax.numpy as jnp
   from jax_supernovae.data import load_hsf_data
   
   # Load raw data
   data = load_hsf_data('19dwz', base_dir='data')
   
   # Convert to JAX arrays
   times = jnp.array(data['time'])
   fluxes = jnp.array(data['flux'])
   fluxerrs = jnp.array(data['fluxerr'])
   zps = jnp.array(data['zp'])

JAX arrays support automatic differentiation, just-in-time (JIT) compilation, and GPU acceleration, making them well-suited for gradient-based optimization and large-scale computations.

Generating Band Indices
--------------------

For efficient flux calculations, JAX-bandflux generates band indices that map each observation to a registered bandpass:

.. code-block:: python

   from jax_supernovae.bandpasses import register_all_bandpasses
   import jax.numpy as jnp
   
   # Register all bandpasses
   bandpass_dict, bridges_dict = register_all_bandpasses()
   
   # Generate band indices
   band_names = data['band']
   band_indices = jnp.array([list(bandpass_dict.keys()).index(band) for band in band_names])

These band indices are used in vectorized flux calculations to select the appropriate bandpass for each observation.

Precomputing Bridge Data
---------------------

For efficient flux calculations, JAX-bandflux precomputes "bridge data" for each bandpass. This bridge data is used in the ``salt3_bandflux`` function to avoid recomputing certain values for each flux calculation:

.. code-block:: python

   from jax_supernovae.salt3 import precompute_bandflux_bridge
   
   # Precompute bridge data for each bandpass
   bridges = [precompute_bandflux_bridge(bandpass_dict[band]) for band in band_names]

The ``load_and_process_data`` function automatically precomputes bridge data for all required bandpasses.

Working with Multiple Supernovae
-----------------------------

You can load and process data for multiple supernovae:

.. code-block:: python

   from jax_supernovae.data import load_and_process_data
   
   # Load and process data for multiple supernovae
   sn_names = ['19dwz', '20aai']
   data_dict = {}
   
   for sn_name in sn_names:
       data_dict[sn_name] = load_and_process_data(
           sn_name=sn_name,
           data_dir='data',
           fix_z=True
       )
   
   # Access data for a specific supernova
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = data_dict['19dwz']

This is useful for joint analyses of multiple supernovae.

Custom Data Formats
----------------

JAX-bandflux is designed to work with the HSF DR1 format, but you can adapt it to work with other data formats by creating custom data loading functions:

.. code-block:: python

   import jax.numpy as jnp
   from astropy.table import Table
   from jax_supernovae.bandpasses import register_all_bandpasses
   from jax_supernovae.salt3 import precompute_bandflux_bridge
   
   def load_custom_data(file_path):
       # Load data from a custom format
       # This is just an example, adapt it to your specific format
       data = Table.read(file_path, format='ascii')
       
       # Register all bandpasses
       bandpass_dict, bridges_dict = register_all_bandpasses()
       
       # Convert to JAX arrays
       times = jnp.array(data['time'])
       fluxes = jnp.array(data['flux'])
       fluxerrs = jnp.array(data['fluxerr'])
       zps = jnp.array(data['zp'])
       
       # Generate band indices
       band_names = data['band']
       band_indices = jnp.array([list(bandpass_dict.keys()).index(band) for band in band_names])
       
       # Precompute bridge data
       bridges = [bridges_dict[band] for band in band_names]
       
       # Load redshift data
       fixed_z = (data['z'][0], data['z_err'][0]) if 'z' in data.colnames else None
       
       return times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z

You can then use this function to load data from your custom format:

.. code-block:: python

   # Load data from a custom format
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_custom_data('path/to/custom/data.txt')

Data Visualization
---------------

You can visualize the data using matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Create a figure
   plt.figure(figsize=(10, 6))
   
   # Plot the data
   unique_bands = np.unique(band_indices)
   colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bands)))
   
   for i, band_idx in enumerate(unique_bands):
       mask = band_indices == band_idx
       plt.errorbar(
           times[mask],
           fluxes[mask],
           yerr=fluxerrs[mask],
           fmt='o',
           color=colors[i],
           label=f'Band {band_idx}'
       )
   
   plt.xlabel('Time (MJD)')
   plt.ylabel('Flux')
   plt.legend()
   plt.title('Supernova Light Curve Data')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

This is useful for inspecting the data before fitting.