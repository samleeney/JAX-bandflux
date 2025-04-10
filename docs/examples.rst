Examples
========

This page provides examples of how to use JAX-bandflux for various tasks related to supernova light curve modeling. Each example includes a description, code snippets, and expected outputs.

.. contents:: Table of Contents
   :local:
   :depth: 1

Optimization with L-BFGS-B
-------------------------

The ``fmin_bfgs.py`` example demonstrates how to use the L-BFGS-B optimization algorithm to fit SALT3 parameters to supernova light curve data.

**Running the Example:**

.. code-block:: bash

   # Download and run the example script
   wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/fmin_bfgs.py
   python fmin_bfgs.py

**What This Example Does:**

1. Loads supernova light curve data for SN 19dwz
2. Registers bandpasses automatically
3. Defines an objective function based on chi-squared
4. Uses L-BFGS-B to find the best-fit SALT3 parameters

**Key Code Sections:**

.. code-block:: python

   # Define the objective function
   def objective(parameters):
       # Create parameter dictionary
       param_dict = {
           'z': fixed_z[0],  # Fixed redshift
           't0': parameters[0],
           'x0': parameters[1],
           'x1': parameters[2],
           'c': parameters[3]
       }
       
       # Calculate model fluxes
       model_fluxes = optimized_salt3_multiband_flux(
           times, bridges, param_dict, zps=zps, zpsys='ab'
       )
       model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
       
       # Calculate chi-squared
       chi2 = float(jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2))
       
       return chi2

**Expected Output:**

.. code-block:: text

   Optimization successful: True
   Number of function evaluations: 87
   
   Best-fit parameters:
            z = 0.152300 (fixed)
           t0 = 58651.234567
           x0 = 1.234567e-05
           x1 = 0.987654
            c = 0.123456
   
   Final chi-squared: 123.45

Nested Sampling
-------------

The ``ns.py`` and ``ns.ipynb`` examples demonstrate how to use nested sampling for Bayesian inference of SALT3 parameters. This approach provides not only parameter estimates but also uncertainties and correlations.

**Running the Example:**

.. code-block:: bash

   # Download and run the example script
   wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/ns.py
   python ns.py

**What This Example Does:**

1. Loads supernova light curve data
2. Registers bandpasses
3. Defines a likelihood function and prior distributions
4. Uses nested sampling to sample the posterior distribution of SALT3 parameters
5. Visualizes the results using corner plots

**Key Code Sections:**

.. code-block:: python

   # Define prior distributions
   prior_dists = {
       't0': distrax.Uniform(low=58500.0, high=58700.0),
       'x0': distrax.Uniform(low=1e-6, high=1e-4),
       'x1': distrax.Uniform(low=-3.0, high=3.0),
       'c': distrax.Uniform(low=-0.3, high=0.3)
   }
   
   # Define log-likelihood function
   @jax.jit
   def compute_loglikelihood(params):
       # Convert parameters to dictionary
       param_dict = {
           'z': fixed_z[0],
           't0': params[0],
           'x0': 10**params[1],  # log parameterization
           'x1': params[2],
           'c': params[3]
       }
       
       # Calculate model fluxes
       model_fluxes = optimized_salt3_multiband_flux(
           times, bridges, param_dict, zps=zps, zpsys='ab'
       )
       model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
       
       # Calculate log-likelihood (Gaussian)
       log_likelihood = -0.5 * jnp.sum(
           ((fluxes - model_fluxes) / fluxerrs)**2 + 
           jnp.log(2 * jnp.pi * fluxerrs**2)
       )
       
       return log_likelihood

Anomaly Detection
---------------

The ``ns_anomaly.py`` example demonstrates how to use nested sampling for anomaly detection in supernova light curves. This approach can identify supernovae that don't fit the standard SALT3 model well.

**Running the Example:**

.. code-block:: bash

   # Download and run the example script
   wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/ns_anomaly.py
   python ns_anomaly.py

**What This Example Does:**

1. Loads supernova light curve data
2. Registers bandpasses
3. Defines a likelihood function and prior
4. Uses nested sampling to sample the posterior distribution of SALT3 parameters
5. Identifies anomalies based on the evidence ratio between the SALT3 model and a more flexible model

**Key Concept: Evidence Ratio**

The evidence ratio (Bayes factor) between two models is calculated as:

.. math::

   R = \frac{P(D|M_1)}{P(D|M_2)}

Where:
* :math:`P(D|M_1)` is the evidence for the SALT3 model
* :math:`P(D|M_2)` is the evidence for a more flexible model

A low evidence ratio indicates that the supernova is not well-described by the SALT3 model.

**Applications:**

* Identifying peculiar supernovae
* Quality control for cosmological analyses
* Discovery of new supernova subtypes

Plotting Light Curves
------------------

The ``plot_light_curve.py`` example demonstrates how to plot supernova light curves with JAX-bandflux. This is useful for visualizing the data and model fits.

**Running the Example:**

.. code-block:: bash

   # Download and run the example script
   wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/plot_light_curve.py
   python plot_light_curve.py

**What This Example Does:**

1. Loads supernova light curve data
2. Registers bandpasses
3. Computes model fluxes using the SALT3 model
4. Plots the observed and model light curves

**Key Code Sections:**

.. code-block:: python

   import matplotlib.pyplot as plt
   from jax_supernovae.salt3 import optimized_salt3_multiband_flux
   
   # Calculate model fluxes
   model_fluxes = optimized_salt3_multiband_flux(
       times, bridges, params, zps=zps, zpsys='ab'
   )
   model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
   
   # Plot the results
   plt.figure(figsize=(10, 6))
   
   # Get unique bands
   unique_bands = jnp.unique(band_indices)
   colors = plt.cm.tab10(jnp.linspace(0, 1, len(unique_bands)))
   
   for i, band_idx in enumerate(unique_bands):
       mask = band_indices == band_idx
       plt.errorbar(
           times[mask] - params['t0'],
           fluxes[mask],
           yerr=fluxerrs[mask],
           fmt='o',
           color=colors[i],
           label=f'Band {band_idx} (Observed)'
       )
       plt.plot(
           times[mask] - params['t0'],
           model_fluxes[mask],
           '-',
           color=colors[i],
           label=f'Band {band_idx} (Model)'
       )

Comparing Models
-------------

The ``plot_comparison.py`` example demonstrates how to compare different supernova models, such as SALT3 and SALT3-NIR. This is useful for understanding the differences between models and their impact on parameter estimation.

**Running the Example:**

.. code-block:: bash

   # Download and run the example script
   wget https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/examples/plot_comparison.py
   python plot_comparison.py

**What This Example Does:**

1. Loads supernova light curve data
2. Registers bandpasses
3. Computes model fluxes using different models (e.g., SALT3, SALT3-NIR)
4. Plots the observed and model light curves for comparison
5. Compares the best-fit parameters from different models

**Key Differences Between Models:**

* **SALT3**: Standard model with coverage from 2800Å to 12000Å
* **SALT3-NIR**: Extended model with coverage from 2800Å to 17000Å
* The NIR extension is particularly important for high-redshift supernovae

Custom Bandpasses
--------------

The ``download_svo_filter.py`` example demonstrates how to download and use custom bandpasses from the Spanish Virtual Observatory (SVO) Filter Profile Service. This is useful for working with data from different instruments or creating synthetic filters.

**Running the Example:**

.. code-block:: bash

   # Download the WFCAM J filter profile
   python examples/download_svo_filter.py --filter UKIRT/WFCAM.J
   
   # List available common filters
   python examples/download_svo_filter.py --list
   
   # Run an example of using a custom bandpass in a SALT3 model fit
   python examples/download_svo_filter.py --example
   
   # Run with a different filter and bandpass name
   python examples/download_svo_filter.py --example --filter 2MASS/2MASS.J --bandpass-name custom_2mass_J
   
   # Create a synthetic WFCAM J filter profile
   python examples/download_svo_filter.py --synthetic
   
   # Customize the number of points in the synthetic profile
   python examples/download_svo_filter.py --synthetic --points 200

**What This Example Does:**

1. Downloads filter profiles from the SVO Filter Profile Service
2. Creates bandpass objects from the downloaded profiles
3. Registers the bandpasses for use in SALT3 model fitting
4. Demonstrates how to use custom bandpasses in a SALT3 model fit

**Available SVO Filters:**

The SVO Filter Profile Service provides access to thousands of filter profiles from various instruments and surveys. Some commonly used filter sets include:

* SDSS (Sloan Digital Sky Survey)
* HST (Hubble Space Telescope)
* JWST (James Webb Space Telescope)
* 2MASS (Two Micron All Sky Survey)
* UKIRT (United Kingdom Infrared Telescope)
* ZTF (Zwicky Transient Facility)
* LSST (Legacy Survey of Space and Time)

Working with Multiple Supernovae
-----------------------------

For population studies and cosmological analyses, you often need to work with multiple supernovae. JAX-bandflux makes this easy with its efficient data loading and processing functions.

**Example Use Cases:**

* Fitting multiple supernovae for cosmological analyses
* Comparing properties across a population of supernovae
* Creating Hubble diagrams
* Studying correlations between supernova parameters

JAX Optimization Techniques
------------------------

JAX provides several optimization techniques that can significantly improve performance:

1. **JIT Compilation**: Using `@jax.jit` to compile functions for faster execution
2. **Vectorization**: Using `jax.vmap` to efficiently process multiple inputs
3. **GPU Acceleration**: Automatically running computations on GPU when available
4. **Precomputed Data**: Using precomputed bridge data for efficient flux calculations

These techniques can provide speedups of 10-100x compared to standard implementations, especially for large datasets or complex models.