Bandpass Loading
==============

What is a Bandpass?
----------------
A bandpass represents the transmission function of an astronomical filter, defining how much light at each wavelength passes through the filter. In supernova light curve modeling, bandpasses are essential for calculating the observed flux in different filters.

Creating a Synthetic Bandpass
--------------------------
.. code-block:: python

   import jax.numpy as jnp
   from jax_supernovae.bandpasses import Bandpass, register_bandpass

   # Create a simple Gaussian bandpass
   wave = jnp.linspace(4000, 6000, 100)  # Wavelength in Angstroms
   center = 5000
   width = 500
   trans = jnp.exp(-((wave - center)**2) / (2 * width**2))

   # Create and register the bandpass
   synthetic_bandpass = Bandpass(wave=wave, trans=trans, name='synthetic_g')
   register_bandpass('synthetic_g', synthetic_bandpass)

Loading a Bandpass from SVO
------------------------
.. code-block:: python

   import os
   import sys
   sys.path.append('examples')
   from download_svo_filter import download_filter
   from jax_supernovae.bandpasses import load_bandpass_from_file, register_bandpass

   # Download a filter profile from SVO
   filter_path = download_filter('SDSS/SDSS.g', output_dir='filter_data')

   # Load and register the bandpass
   sdss_g_bandpass = load_bandpass_from_file(filter_path)
   register_bandpass('sdss_g', sdss_g_bandpass)

Using Built-in Bandpasses
----------------------
.. code-block:: python

   from jax_supernovae.bandpasses import register_all_bandpasses

   # Register all built-in bandpasses
   bandpass_dict, bridges_dict = register_all_bandpasses()

   # Available bandpasses include:
   # - ZTF: 'ztfg', 'ztfr'
   # - SDSS: 'g', 'r', 'i', 'z'
   # - ATLAS: 'c', 'o'
   # - 2MASS: 'H'
   # - WFCAM: 'J', 'J_1D3'

   # Access a specific bandpass
   ztfg_bandpass = bandpass_dict['ztfg']