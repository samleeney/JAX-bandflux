API Reference
============

This section provides detailed API documentation for JAX-bandflux. The API is organized into several modules, each focusing on a specific aspect of supernova light curve modeling.

Overview
-------

JAX-bandflux's API is designed to be modular, efficient, and easy to use. The main modules are:

* **salt3**: Implementation of the SALT3 model for supernova light curves
* **bandpasses**: Management of bandpass filters for different instruments
* **data**: Loading and processing supernova light curve data
* **utils**: Utility functions for various tasks
* **constants**: Physical constants used throughout the codebase

Module Relationships
------------------

The modules are designed to work together seamlessly:

* The **salt3** module uses the **bandpasses** module to calculate model fluxes through different filters
* The **data** module loads data and registers the necessary **bandpasses**
* The **utils** module provides helper functions used by all other modules
* The **constants** module defines physical constants used by the **salt3** and **bandpasses** modules

Using the API
-----------

Here's a typical workflow using the JAX-bandflux API:

1. Load data using functions from the **data** module
2. Register bandpasses using the **bandpasses** module
3. Calculate model fluxes using the **salt3** module
4. Optimize parameters or perform Bayesian inference

For example:

.. code-block:: python

   from jax_supernovae.data import load_and_process_data
   from jax_supernovae.salt3 import optimized_salt3_multiband_flux
   
   # Load data (bandpasses are registered automatically)
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz', data_dir='data', fix_z=True
   )
   
   # Define parameters
   params = {
       'z': fixed_z[0],
       't0': 58650.0,
       'x0': 1e-5,
       'x1': 0.0,
       'c': 0.0
   }
   
   # Calculate model fluxes
   model_fluxes = optimized_salt3_multiband_flux(
       times, bridges, params, zps=zps, zpsys='ab'
   )

.. toctree::
   :maxdepth: 2
   
   salt3
   bandpasses
   data
   utils
   constants