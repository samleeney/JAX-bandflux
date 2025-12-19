Installation
===========

This section provides essential instructions for installing JAX-bandflux.

CPU vs CUDA wheels
------------------

JAX-bandflux does **not** force a CUDA dependency. Choose the JAX wheel that matches your hardware:

* **CPU**:

  .. code-block:: bash

     pip install jax-bandflux
     pip install --upgrade "jax[cpu]"

* **CUDA** (example for CUDA 12):

  .. code-block:: bash

     pip install jax-bandflux
     pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  For other CUDA versions, see the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ and pick the matching wheel for your driver/toolkit.

Dependencies
------------

* Python 3.10+
* JAX 0.4.20+
* NumPy 1.24.0+
* SNCosmo 2.9.0+
* Astropy 5.0.0+

Verification
------------

To verify that JAX-bandflux is installed correctly:

.. code-block:: bash

   python -c "import jax_supernovae; print('JAX-bandflux successfully installed')"

This command should display a success message if JAX-bandflux is installed correctly.
