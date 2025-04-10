Utils Module
============

.. automodule:: jax_supernovae.utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``utils`` module provides utility functions for JAX-bandflux. It includes functions for interpolation, integration, and other common operations used throughout the package.

Key Functions
------------

.. autosummary::
   :nosignatures:

   jax_supernovae.utils.trapz
   jax_supernovae.utils.interp1d
   jax_supernovae.utils.interp2d

Example Usage
------------

.. code-block:: python

   import jax.numpy as jnp
   from jax_supernovae.utils import trapz, interp1d, interp2d
   
   # Trapezoidal integration
   x = jnp.linspace(0, 1, 100)
   y = x**2
   integral = trapz(y, x)
   
   # 1D interpolation
   x_data = jnp.array([0, 1, 2, 3, 4])
   y_data = jnp.array([0, 1, 4, 9, 16])
   x_interp = jnp.linspace(0, 4, 100)
   y_interp = interp1d(x_data, y_data, x_interp)
   
   # 2D interpolation
   x_data = jnp.array([0, 1, 2, 3, 4])
   y_data = jnp.array([0, 1, 2, 3, 4])
   z_data = jnp.outer(x_data, y_data)
   x_interp = jnp.linspace(0, 4, 100)
   y_interp = jnp.linspace(0, 4, 100)
   z_interp = interp2d(x_data, y_data, z_data, x_interp, y_interp)