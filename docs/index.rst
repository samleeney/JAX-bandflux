JAX-bandflux Documentation
=========================

JAX-bandflux: Differentiable Supernova Light Curve Modeling
----------------------------------------------------------

.. image:: https://badge.fury.io/py/jax-bandflux.svg
    :target: https://badge.fury.io/py/jax-bandflux
    :alt: PyPI version

JAX-bandflux is a Python package that implements supernova light curve modeling using JAX. The codebase offers a differentiable approach to core SNCosmo functionality, enabling efficient gradient-based optimization and GPU acceleration for supernova cosmology research.

Why JAX-bandflux?
----------------

JAX-bandflux provides several key advantages over traditional supernova light curve modeling frameworks:

* **Differentiability**: By implementing the SALT3 model in JAX, JAX-bandflux enables automatic differentiation of the entire modeling pipeline, allowing for efficient gradient-based optimization.

* **Performance**: JAX's just-in-time (JIT) compilation and GPU acceleration provide significant performance improvements, especially for large-scale analyses involving many supernovae.

* **Flexibility**: The modular design allows for easy customization of bandpasses, models, and optimization strategies.

* **Compatibility**: JAX-bandflux maintains compatibility with existing SNCosmo data formats and models, making it easy to integrate into existing workflows.

* **Research-Friendly**: The codebase is designed with research in mind, providing tools for both standard analyses and novel approaches to supernova cosmology.

Key Features
-----------

* Differentiable implementation of SALT3 model for supernova light curves
* GPU-accelerated flux calculations using JAX
* Flexible bandpass management for various astronomical filters
* Efficient data loading and processing routines
* Support for gradient-based optimization and nested sampling
* Comprehensive documentation and examples

Package Structure
---------------

JAX-bandflux is organized into several key components:

.. code-block:: text

    jax_supernovae/
    ├── salt3.py         # SALT3 model implementation
    ├── bandpasses.py    # Bandpass management
    ├── data.py          # Data loading and processing
    ├── utils.py         # Utility functions
    ├── constants.py     # Physical constants
    ├── data/            # Example data files
    └── sncosmo-modelfiles/ # Model and bandpass files

The following diagram shows the relationships between key components:

.. raw:: html

    <div class="mermaid">
    graph TD
        A[JAX-bandflux] --> B[SALT3 Model]
        A --> C[Bandpass Management]
        A --> D[Data Handling]
        A --> E[Optimization Methods]
        B --> F[salt3_bandflux]
        B --> G[salt3_flux]
        C --> H[Built-in Bandpasses]
        C --> I[Custom Bandpasses]
        C --> J[SVO Integration]
        D --> K[HSF Data Format]
        D --> L[Redshift Handling]
        E --> M[L-BFGS-B]
        E --> N[Nested Sampling]
    </div>

    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({startOnLoad:true});</script>

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   data_loading
   bandpass_loading
   model_fluxes
   dust
   sampling

Quick Links
----------

* :doc:`installation` - Installation instructions
* :doc:`quickstart` - Get started quickly with basic examples
* :doc:`data_loading` - Learn how to load and process supernova data
* :doc:`bandpass_loading` - Working with astronomical filters and bandpasses
* :doc:`model_fluxes` - Computing model fluxes using the SALT3 model
* :doc:`dust` - Applying dust extinction to supernova models
* :doc:`sampling` - Techniques for parameter estimation and sampling

Getting Help
-----------

If you encounter any issues or have questions about JAX-bandflux, please:

* Check the :doc:`installation` and :doc:`quickstart` guides
* Review the documentation sections for specific functionality
* Look for similar issues in the `GitHub repository <https://github.com/samleeney/JAX-bandflux/issues>`_
* Open a new issue if your problem hasn't been addressed

Indices and tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`