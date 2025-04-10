Installation
===========

This page provides comprehensive instructions for installing JAX-bandflux and setting up your environment for supernova light curve modeling.

Requirements
-----------

JAX-bandflux requires:

* Python 3.8 or later
* JAX 0.4.20 or later
* NumPy 1.24.0 or later
* SNCosmo 2.9.0 or later
* Astropy 5.0.0 or later
* Matplotlib 3.5.0 or later (for visualization)

Basic Installation
----------------

You can install JAX-bandflux using pip:

.. code-block:: bash

   pip install jax-bandflux

This will install JAX-bandflux and its dependencies.

Development Installation
----------------------

For development, you can install JAX-bandflux from source:

.. code-block:: bash

   git clone https://github.com/samleeney/JAX-bandflux.git
   cd JAX-bandflux
   pip install -e .[dev]

This will install JAX-bandflux in development mode, along with additional development dependencies like pytest, pytest-cov, black, and isort.

GPU Support
----------

JAX-bandflux can leverage GPU acceleration through JAX, which can significantly speed up computations, especially for large-scale analyses. To use GPU acceleration, you need to install the GPU version of JAX.

Prerequisites for GPU Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before installing JAX with GPU support, ensure you have:

1. A CUDA-compatible GPU
2. CUDA Toolkit (11.1 or later recommended)
3. cuDNN (compatible with your CUDA version)

Installing JAX with GPU Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install JAX with GPU support:

.. code-block:: bash

   pip install --upgrade pip
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

This will install the appropriate version of JAX for your CUDA installation.

Verifying GPU Availability
^^^^^^^^^^^^^^^^^^^^^^^^

To verify that JAX can access your GPU:

.. code-block:: python

   import jax
   print("Available devices:", jax.devices())
   print("Default device:", jax.default_backend())

You should see your GPU listed in the available devices.

Note: If you're using a Jupyter notebook, you may need to restart the kernel after installing JAX with GPU support.

Optional Dependencies
-------------------

For advanced functionality, you may want to install these optional dependencies:

.. code-block:: bash

   # For nested sampling
   pip install blackjax anesthetic

   # For progress bars
   pip install tqdm

   # For interactive visualizations
   pip install corner

Verifying Installation
--------------------

To verify that JAX-bandflux is installed correctly, you can run the following command:

.. code-block:: bash

   python -c "import jax_supernovae; print(jax_supernovae.__version__)"

This should print the version of JAX-bandflux that you have installed.

You can also run the tests to ensure everything is working correctly:

.. code-block:: bash

   python -m pytest tests/

Running a Simple Example
----------------------

To verify that everything is working properly, you can run a simple example:

.. code-block:: bash

   # Download an example script
   wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/fmin_bfgs.py
   
   # Run the example
   python fmin_bfgs.py

This should fit a SALT3 model to a supernova light curve and display the results.

Troubleshooting
-------------

Common Issues
^^^^^^^^^^^

1. **ImportError: No module named 'jax_supernovae'**

   Make sure you have installed JAX-bandflux correctly. Try reinstalling:
   
   .. code-block:: bash
   
      pip uninstall jax-bandflux
      pip install jax-bandflux

2. **JAX GPU acceleration not working**

   Verify your CUDA installation:
   
   .. code-block:: bash
   
      nvidia-smi  # Check if GPU is detected
      python -c "import jax; print(jax.devices())"  # Check if JAX can see the GPU
   
   If JAX doesn't detect your GPU, make sure your CUDA and cuDNN versions are compatible with your JAX version.

3. **Missing data files**

   If you encounter errors about missing data files, you may need to download them:
   
   .. code-block:: bash
   
      # Create data directory if it doesn't exist
      mkdir -p data/19dwz
      
      # Download example data files
      wget -O data/19dwz/all.phot https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/jax_supernovae/data/19dwz/all.phot
      wget -O data/redshifts.dat https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/jax_supernovae/data/redshifts.dat

Getting Help
^^^^^^^^^^

If you encounter issues not covered here, please:

1. Check the `GitHub repository <https://github.com/samleeney/JAX-bandflux/issues>`_ for similar issues
2. Open a new issue with details about your problem
3. Include information about your environment (Python version, JAX version, etc.)