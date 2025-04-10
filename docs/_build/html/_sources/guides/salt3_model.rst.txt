SALT3 Model
==========

This guide provides in-depth information about the SALT3 model implementation in JAX-bandflux.

Introduction to SALT3
------------------

SALT3 (Spectral Adaptive Lightcurve Template 3) is a parametric model for Type Ia supernova light curves. It extends the SALT2 model with improved near-infrared coverage and updated training data. The model is described in detail in Kenworthy et al. (2021) and Pierel et al. (2022).

The SALT model is of the form:

.. math::

   F(p, \lambda) = x_0 \left[ M_0(p, \lambda) + x_1 M_1(p, \lambda) + \ldots \right] \times \exp \left[ c \times CL(\lambda) \right]

Where:

* :math:`x_0` is the overall flux normalization
* :math:`x_1` is the stretch parameter
* :math:`t_0` is the time of peak brightness
* :math:`c` is the color parameter
* :math:`M_0(p, \lambda)` and :math:`M_1(p, \lambda)` are functions that describe the underlying flux surfaces
* :math:`p` is a function of redshift and :math:`t-t_0`
* :math:`CL(\lambda)` is the color law

JAX-bandflux implements the SALT3 model in a differentiable way using JAX, which enables efficient gradient-based optimization and GPU acceleration.

SALT3 Parameters
--------------

The SALT3 model has five main parameters:

1. ``z``: Redshift of the supernova
2. ``t0``: Time of peak brightness (in MJD)
3. ``x0``: Overall flux normalization
4. ``x1``: Stretch parameter (related to the width of the light curve)
5. ``c``: Color parameter (related to the reddening of the supernova)

These parameters are typically stored in a dictionary:

.. code-block:: python

   params = {
       'z': 0.1,
       't0': 0.0,
       'x0': 1e-5,
       'x1': 0.0,
       'c': 0.0
   }

Model Files
---------

JAX-bandflux includes SALT3 model files in the ``sncosmo-modelfiles/models`` directory. Three model variants are available:

- ``salt3-nir``: Extended SALT3 model with near-infrared coverage (2800-17000Å)
- ``salt3``: Standard SALT3 model (2800-12000Å)

Each model directory contains the following key files:

- ``salt3_template_0.dat``: M0 component (mean SN Ia spectrum)
- ``salt3_template_1.dat``: M1 component (spectral variation)
- ``salt3_color_correction.dat``: Colour law coefficients
- ``SALT3.INFO``: Model metadata and configuration
- Additional files for variance and covariance

Computing Model Flux
-----------------

JAX-bandflux provides functions for computing model flux and bandflux:

.. code-block:: python

   from jax_supernovae.salt3 import salt3_flux, salt3_bandflux
   import jax.numpy as jnp
   
   # Define parameters
   params = {
       'z': 0.1,
       't0': 0.0,
       'x0': 1e-5,
       'x1': 0.0,
       'c': 0.0
   }
   
   # Compute model flux at a specific time and wavelength
   time = 0.0
   wavelength = jnp.linspace(4000, 5000, 100)
   flux = salt3_flux(time, wavelength, params)
   
   # Compute bandflux at a specific time for a bandpass
   from jax_supernovae.bandpasses import register_all_bandpasses
   bandpass_dict, bridges_dict = register_all_bandpasses()
   
   time = 0.0
   bandpass = bandpass_dict['ztfg']
   bridge = bridges_dict['ztfg']
   
   # Using the bandpass directly
   flux = salt3_bandflux(time, bandpass, params)
   
   # Using precomputed bridge data (more efficient)
   flux = salt3_bandflux(time, bridge, params)

The ``salt3_flux`` function computes the model flux at a specific time and wavelength, while the ``salt3_bandflux`` function computes the integrated flux through a bandpass.

Bandflux Computation
-----------------

The computation of the bandflux is achieved by integrating the model flux across the applied bandpass filters:

.. math::

   \text{bandflux} = \int_{\lambda_\text{min}}^{\lambda_\text{max}} F(\lambda) \cdot T(\lambda) \cdot \frac{\lambda}{hc} \, d\lambda

Where:

* :math:`F(\lambda)` is the model flux as a function of wavelength
* :math:`T(\lambda)` is the transmission function of the bandpass filter
* :math:`\lambda` is the wavelength
* :math:`h` is the Planck constant
* :math:`c` is the speed of light

For efficiency, JAX-bandflux precomputes "bridge data" for each bandpass using the ``precompute_bandflux_bridge`` function:

.. code-block:: python

   from jax_supernovae.salt3 import precompute_bandflux_bridge
   
   # Precompute bridge data
   bridge = precompute_bandflux_bridge(bandpass)
   
   # Use the bridge data in flux calculations
   flux = salt3_bandflux(time, bridge, params)

The ``register_all_bandpasses`` function automatically precomputes bridge data for all registered bandpasses.

Differentiable Implementation
--------------------------

JAX-bandflux implements the SALT3 model in a differentiable way using JAX. This enables efficient gradient-based optimization and GPU acceleration.

For example, you can compute the gradient of the bandflux with respect to the parameters:

.. code-block:: python

   import jax
   
   # Define a function that computes the bandflux
   def compute_bandflux(params_array):
       # Convert the array to a dictionary
       params = {
           'z': params_array[0],
           't0': params_array[1],
           'x0': params_array[2],
           'x1': params_array[3],
           'c': params_array[4]
       }
       
       # Compute the bandflux
       return salt3_bandflux(time, bridge, params)
   
   # Compute the gradient
   params_array = jnp.array([0.1, 0.0, 1e-5, 0.0, 0.0])
   gradient = jax.grad(compute_bandflux)(params_array)
   
   print("Gradient:", gradient)

This gradient can be used in optimization algorithms like L-BFGS-B to find the best-fit parameters.

JIT Compilation
------------

JAX-bandflux uses JAX's just-in-time (JIT) compilation to improve performance. You can JIT-compile functions that use the SALT3 model:

.. code-block:: python

   import jax
   
   # Define a function that computes the chi-squared
   def compute_chi2(params_array, times, fluxes, fluxerrs, bridges, zps):
       # Convert the array to a dictionary
       params = {
           'z': params_array[0],
           't0': params_array[1],
           'x0': params_array[2],
           'x1': params_array[3],
           'c': params_array[4]
       }
       
       # Compute model fluxes
       model_fluxes = salt3_bandflux(times, bridges, params, zp=zps)
       
       # Compute chi-squared
       chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
       
       return chi2
   
   # JIT-compile the function
   jit_compute_chi2 = jax.jit(compute_chi2)
   
   # Use the JIT-compiled function
   params_array = jnp.array([0.1, 0.0, 1e-5, 0.0, 0.0])
   chi2 = jit_compute_chi2(params_array, times, fluxes, fluxerrs, bridges, zps)
   
   print("Chi-squared:", chi2)

JIT compilation can significantly improve performance, especially for repeated evaluations of the same function.

Vectorization
-----------

JAX-bandflux uses JAX's vectorization capabilities to compute fluxes for multiple times, wavelengths, or parameters in parallel:

.. code-block:: python

   import jax
   
   # Define a function that computes the bandflux for a single time
   def compute_bandflux_single(time, bridge, params):
       return salt3_bandflux(time, bridge, params)
   
   # Vectorize the function over times
   compute_bandflux_vectorized = jax.vmap(compute_bandflux_single, in_axes=(0, None, None))
   
   # Compute bandfluxes for multiple times
   times = jnp.linspace(-10, 30, 100)
   fluxes = compute_bandflux_vectorized(times, bridge, params)
   
   print("Fluxes shape:", fluxes.shape)

Vectorization can significantly improve performance for large-scale computations.

GPU Acceleration
-------------

JAX-bandflux can leverage GPU acceleration through JAX. To use GPU acceleration, you need to install the GPU version of JAX. Please refer to the `JAX installation guide <https://github.com/google/jax#installation>`_ for detailed instructions.

Once you have installed the GPU version of JAX, you can use JAX-bandflux as usual, and JAX will automatically use the GPU for computations.

Model Variants
-----------

JAX-bandflux supports different variants of the SALT3 model:

1. **SALT3**: The standard SALT3 model with wavelength coverage from 2800Å to 12000Å.
2. **SALT3-NIR**: An extended version of SALT3 with near-infrared coverage from 2800Å to 17000Å.

You can specify which model to use when loading data:

.. code-block:: python

   from jax_supernovae.data import load_and_process_data
   
   # Load and process data with the SALT3-NIR model
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',
       data_dir='data',
       fix_z=True,
       model_name='salt3-nir'  # Use the SALT3-NIR model
   )

The default model is ``salt3-nir``.

Custom Models
-----------

You can use custom SALT3 models by placing the model files in a subdirectory of ``sncosmo-modelfiles/models``. The model files should follow the same structure as the built-in models.

For example, to use a custom model called ``my-salt3``:

1. Create a directory ``sncosmo-modelfiles/models/my-salt3/my-salt3-version``
2. Place the model files in this directory
3. Specify the model name when loading data:

.. code-block:: python

   from jax_supernovae.data import load_and_process_data
   
   # Load and process data with the custom model
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',
       data_dir='data',
       fix_z=True,
       model_name='my-salt3'  # Use the custom model
   )