Installation
===========

This section provides essential instructions for installing JAX-bandflux. Python
3.10+ is required. Core dependencies include JAX (0.4.20+), NumPy (1.24.0+),
Astropy (5.0+), and SNCosmo (2.9.0+). SALT3/SALT3-NIR model files ship with the
package; no extra download is needed.

Basic Installation
-----------------

Install JAX-bandflux using pip:

.. code-block:: bash

   pip install jax-bandflux

GPU/CUDA wheels
---------------

To run on NVIDIA GPUs, install the matching CUDA wheel for JAX before or with
jax-bandflux. For CUDA 12:

.. code-block:: bash

   pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   pip install jax-bandflux

or in one go:

.. code-block:: bash

   pip install "jax-bandflux[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

See the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
for other CUDA versions.

Nested sampling extras
----------------------

Optional dependencies for the nested sampling examples:

.. code-block:: bash

   pip install "jax-bandflux[nested]"

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/samleeney/JAX-bandflux.git
   cd JAX-bandflux
   pip install -e ".[dev,nested,docs]"

Verification
-----------

To verify that JAX-bandflux is installed correctly:

.. code-block:: bash

   python -c "import jax_supernovae; print('JAX-bandflux successfully installed')"

This command should display a success message if JAX-bandflux is installed correctly.
