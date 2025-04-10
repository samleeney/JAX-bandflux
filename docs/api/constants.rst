Constants Module
===============

.. automodule:: jax_supernovae.constants
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``constants`` module provides physical constants and conversion factors used throughout JAX-bandflux. These constants are used in the computation of model fluxes and bandfluxes.

Key Constants
------------------

.. autosummary::
   :nosignatures:

   jax_supernovae.constants.H_ERG_S
   jax_supernovae.constants.C_AA_PER_S
   jax_supernovae.constants.HC_ERG_AA

Example Usage
------------

.. code-block:: python

   import jax.numpy as jnp
   from jax_supernovae.constants import HC_ERG_AA
   
   # Convert wavelength to energy
   wavelength = jnp.linspace(4000, 5000, 100)  # Angstroms
   energy = HC_ERG_AA / wavelength  # ergs