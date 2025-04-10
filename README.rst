=======================================
JAX Bandflux for Supernovae SALT3 model fitting
=======================================
:Author: Samuel Alan Kossoff Leeney
:version: |pypi version|
:Homepage: https://github.com/samleeney/JAX-bandflux
:Documentation: https://jax-bandflux.readthedocs.io/

.. |PyPI version| image:: https://badge.fury.io/py/jax-bandflux.svg
   :target: https://badge.fury.io/py/jax-bandflux
   :alt: PyPI version

.. image:: https://github.com/samleeney/JAX-bandflux/workflows/Tests/badge.svg
   :target: https://github.com/samleeney/JAX-bandflux/actions
   :alt: Build Status

`JAX-bandflux` presents an implementation of supernova light curve modelling using JAX. The codebase offers a differentiable approach to core SNCosmo functionality implemented in JAX.

Quickstart
----------

Run example analagous to SNCosmo's `Using a custom fitter` example:

.. code:: bash

   pip install jax-bandflux
   wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/fmin_bfgs.py
   python fmin_bfgs.py

Testing
-------

This repository implements the JAX version of the SNCosmo bandflux function. Although the implementations are nearly identical, a minor difference exists due to the absence of a specific interpolation function in JAX, resulting in a discrepancy of approximately 0.001% in bandflux results. Tests have been provided to confirm that key functions in the SNCosmo version correspond with our JAX implementation. It is recommended to run these tests, especially following any modifications.

What is the .airules file?
--------------------------

.airules
========

The `.airules` file provides essential context to help language models understand 
and work with this codebase—particularly for new code that may not be included 
in model training datasets. It contains detailed information on:

- Data structures  
- Core functions  
- Implementation constraints  
- Testing requirements  

If you are using `cursor`, rename this file to `.cursorrules` to enable 
automatic context integration.
