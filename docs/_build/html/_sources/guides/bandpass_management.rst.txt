Bandpass Management
=================

This guide provides in-depth information about bandpass management in JAX-bandflux.

Introduction
-----------

In supernova light curve modeling, bandpasses represent the transmission functions of astronomical filters. They define how much light at each wavelength passes through the filter. JAX-bandflux provides a flexible system for managing bandpasses, including built-in support for common astronomical filters and the ability to register custom bandpasses.

The Bandpass Class
---------------

The core of bandpass management in JAX-bandflux is the ``Bandpass`` class:

.. code-block:: python

   from jax_supernovae.bandpasses import Bandpass
   import jax.numpy as jnp
   
   # Create a bandpass
   wave = jnp.linspace(4000, 5000, 100)  # Wavelength in Angstroms
   trans = jnp.ones_like(wave)  # Transmission
   bandpass = Bandpass(wave=wave, trans=trans)

The ``Bandpass`` class has the following attributes:

- ``wave``: A JAX array of wavelengths in Angstroms
- ``trans``: A JAX array of transmission values (between 0 and 1)
- ``name``: An optional name for the bandpass
- ``meta``: An optional dictionary of metadata

Built-in Bandpasses
----------------

JAX-bandflux includes built-in support for several common astronomical filters:

- ZTF bandpasses: ``ztfg``, ``ztfr``
- ATLAS bandpasses: ``c``, ``o``
- SDSS bandpasses: ``g``, ``r``, ``i``, ``z``
- 2MASS bandpasses: ``H``
- WFCAM bandpasses: ``J``, ``J_1D3``

You can register all built-in bandpasses using the ``register_all_bandpasses`` function:

.. code-block:: python

   from jax_supernovae.bandpasses import register_all_bandpasses
   
   # Register all built-in bandpasses
   bandpass_dict, bridges_dict = register_all_bandpasses()
   
   # Access a specific bandpass
   ztfg_bandpass = bandpass_dict['ztfg']

The ``register_all_bandpasses`` function returns two dictionaries:

1. ``bandpass_dict``: A dictionary mapping bandpass names to ``Bandpass`` objects
2. ``bridges_dict``: A dictionary mapping bandpass names to precomputed bridge data for efficient flux calculations

Registering Custom Bandpasses
--------------------------

You can register custom bandpasses using the ``register_bandpass`` function:

.. code-block:: python

   from jax_supernovae.bandpasses import register_bandpass
   
   # Register a bandpass
   register_bandpass('my_bandpass', bandpass)

This adds the bandpass to the global registry, making it available for use in flux calculations.

Loading Bandpasses from Files
--------------------------

JAX-bandflux provides functions for loading bandpasses from files:

.. code-block:: python

   from jax_supernovae.bandpasses import load_bandpass_from_file
   
   # Load a bandpass from a file
   bandpass = load_bandpass_from_file('path/to/bandpass.dat')
   
   # Register the bandpass
   register_bandpass('my_bandpass', bandpass)

The file should be in a simple two-column format:

.. code-block::

   wavelength1 transmission1
   wavelength2 transmission2
   ...

Where:

- ``wavelength`` is in Angstroms
- ``transmission`` is a value between 0 and 1

Using the SVO Filter Profile Service
---------------------------------

JAX-bandflux includes a utility script to download filter profiles from the Spanish Virtual Observatory (SVO) Filter Profile Service:

.. code-block:: bash

   # Download the WFCAM J filter profile
   python examples/download_svo_filter.py --filter UKIRT/WFCAM.J
   
   # List available common filters
   python examples/download_svo_filter.py --list

You can also use this script programmatically:

.. code-block:: python

   import os
   import sys
   sys.path.append('examples')
   from download_svo_filter import download_filter
   
   # Download a filter profile
   filter_path = download_filter('UKIRT/WFCAM.J', output_dir='filter_data')
   
   # Load the bandpass
   from jax_supernovae.bandpasses import load_bandpass_from_file, register_bandpass
   bandpass = load_bandpass_from_file(filter_path)
   
   # Register the bandpass
   register_bandpass('WFCAM_J', bandpass)

Creating Synthetic Bandpasses
--------------------------

You can create synthetic bandpasses using the ``create_synthetic_filter`` function in the ``download_svo_filter.py`` script:

.. code-block:: python

   import sys
   sys.path.append('examples')
   from download_svo_filter import create_synthetic_filter
   
   # Create a synthetic filter profile
   filter_path = create_synthetic_filter(
       central_wavelength=12350,  # Central wavelength in Angstroms
       fwhm=1545,  # Full width at half maximum in Angstroms
       points=100,  # Number of points in the profile
       output_file='filter_data/synthetic_J.dat'  # Output file path
   )
   
   # Load the synthetic bandpass
   from jax_supernovae.bandpasses import load_bandpass_from_file, register_bandpass
   bandpass = load_bandpass_from_file(filter_path)
   
   # Register the bandpass
   register_bandpass('synthetic_J', bandpass)

Precomputing Bridge Data
---------------------

For efficient flux calculations, JAX-bandflux precomputes "bridge data" for each bandpass. This bridge data is used in the ``salt3_bandflux`` function to avoid recomputing certain values for each flux calculation.

.. code-block:: python

   from jax_supernovae.salt3 import precompute_bandflux_bridge
   
   # Precompute bridge data
   bridge = precompute_bandflux_bridge(bandpass)
   
   # Use the bridge data in flux calculations
   from jax_supernovae.salt3 import salt3_bandflux
   
   # Define SALT3 parameters
   params = {
       'z': 0.1,
       't0': 0.0,
       'x0': 1e-5,
       'x1': 0.0,
       'c': 0.0
   }
   
   # Compute bandflux
   time = 0.0
   flux = salt3_bandflux(time, bridge, params)

The ``register_all_bandpasses`` function automatically precomputes bridge data for all registered bandpasses.

Custom Bandpass Configuration
--------------------------

You can configure custom bandpasses in your ``settings.yaml`` file:

.. code-block:: yaml

   # As a list of file paths
   custom_bandpass_files:
     - '/path/to/custom_bandpass1.dat'
     - '/path/to/custom_bandpass2.dat'
   
   # Or as a dictionary mapping names to file paths
   custom_bandpass_files:
     custom_band1: '/path/to/custom_bandpass1.dat'
     custom_band2: '/path/to/custom_bandpass2.dat'
   
   # Specify which bandpasses to use
   selected_bandpasses: ['g', 'r', 'ztfg', 'ztfr', 'c', 'o', 'custom_band1']

JAX-bandflux will automatically load and register these bandpasses when you call ``register_all_bandpasses``.

Bandpass Visualization
-------------------

You can visualize bandpasses using matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot the bandpass
   plt.figure(figsize=(10, 6))
   plt.plot(bandpass.wave, bandpass.trans)
   plt.xlabel('Wavelength (Angstroms)')
   plt.ylabel('Transmission')
   plt.title('Bandpass Transmission')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

This is useful for verifying that bandpasses are loaded correctly and for understanding the wavelength coverage of different filters.