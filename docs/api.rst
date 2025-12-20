API Reference
=============

Core classes and functions are documented here via autodoc. These are the primary entry points for users; internal helper modules are omitted.

Sources
-------

.. automodule:: jax_supernovae.source
   :members: SALT3Source, TimeSeriesSource
   :undoc-members:
   :show-inheritance:

Bandpasses
----------

.. automodule:: jax_supernovae.bandpasses
   :members: Bandpass, register_all_bandpasses, register_bandpass, get_bandpass, load_bandpass, load_custom_bandpasses
   :undoc-members:
   :show-inheritance:

Data utilities
--------------

.. automodule:: jax_supernovae.data
   :members: load_and_process_data, load_hsf_data, load_redshift, get_all_supernovae_with_redshifts
   :undoc-members:
   :show-inheritance:

Salt3 helpers
-------------

.. automodule:: jax_supernovae.salt3
   :members: precompute_bandflux_bridge, optimized_salt3_bandflux, optimized_salt3_multiband_flux
   :undoc-members:
   :show-inheritance:

