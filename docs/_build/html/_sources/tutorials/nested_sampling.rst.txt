Nested Sampling Tutorial
=====================

This tutorial demonstrates how to use nested sampling with JAX-bandflux for Bayesian inference of SALT3 parameters.

Prerequisites
-----------

Before starting this tutorial, make sure you have installed JAX-bandflux and the anesthetic package for nested sampling:

.. code-block:: bash

   pip install jax-bandflux anesthetic

Introduction to Nested Sampling
----------------------------

Nested sampling is a computational approach for Bayesian inference that is particularly useful for parameter estimation and model comparison. Unlike optimization methods that find the maximum likelihood, nested sampling explores the entire parameter space and computes the Bayesian evidence (marginal likelihood).

The key advantages of nested sampling include:

1. It provides posterior distributions for all parameters
2. It computes the Bayesian evidence, which can be used for model comparison
3. It can handle multimodal posteriors and complex parameter spaces

Loading Data
-----------

First, let's load some supernova light curve data:

.. code-block:: python

   from jax_supernovae.data import load_and_process_data
   
   # Load and process data with automatic bandpass registration
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',  # Name of the supernova
       data_dir='data',  # Optional, the default is 'data'
       fix_z=True        # Whether to load and fix redshift from redshifts.dat
   )

Defining the Likelihood and Prior
------------------------------

For nested sampling, we need to define a likelihood function and a prior:

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np
   from jax_supernovae.salt3 import salt3_bandflux
   
   def log_likelihood(parameters):
       # Create a dictionary containing parameters
       params = {
           'z': fixed_z[0] if fixed_z is not None else parameters[0],
           't0': parameters[0 if fixed_z is not None else 1],
           'x0': parameters[1 if fixed_z is not None else 2],
           'x1': parameters[2 if fixed_z is not None else 3],
           'c': parameters[3 if fixed_z is not None else 4]
       }
       
       # Compute model fluxes for all observations
       model_fluxes = salt3_bandflux(times, bridges, params, zp=zps)
       
       # Calculate the log-likelihood (assuming Gaussian errors)
       chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
       log_like = -0.5 * chi2
       
       return float(log_like)
   
   def prior_transform(unit_cube):
       """Transform from the unit cube to the parameter space."""
       # Define the parameter ranges
       if fixed_z is not None:
           # t0, x0, x1, c
           ranges = [
               (-10, 10),       # t0
               (1e-10, 1e-2),   # x0
               (-3, 3),         # x1
               (-1, 1)          # c
           ]
       else:
           # z, t0, x0, x1, c
           ranges = [
               (0.01, 0.2),     # z
               (-10, 10),       # t0
               (1e-10, 1e-2),   # x0
               (-3, 3),         # x1
               (-1, 1)          # c
           ]
       
       # Transform from unit cube to parameter space
       params = np.zeros_like(unit_cube)
       for i, (lower, upper) in enumerate(ranges):
           if i == 2:  # x0 (log-uniform)
               params[i] = 10**(np.log10(lower) + unit_cube[i] * (np.log10(upper) - np.log10(lower)))
           else:  # uniform
               params[i] = lower + unit_cube[i] * (upper - lower)
       
       return params

Running Nested Sampling
--------------------

Now, let's run nested sampling using the anesthetic package:

.. code-block:: python

   import nestle
   
   # Number of parameters
   n_params = 4 if fixed_z is not None else 5
   
   # Run nested sampling
   result = nestle.sample(
       log_likelihood,
       prior_transform,
       n_params,
       method='multi',
       npoints=1000,
       callback=lambda info: print(f"Iteration {info['it']}, log(Z) = {info['logz']:.2f}")
   )
   
   # Print the results
   print("Nested sampling completed!")
   print(f"Log evidence: {result.logz:.2f} ± {result.logzerr:.2f}")
   
   # Extract posterior samples
   weights = np.exp(result.logwt - result.logz)
   samples = result.samples
   
   # Compute weighted mean and standard deviation
   mean = np.sum(samples * weights[:, np.newaxis], axis=0)
   var = np.sum(weights[:, np.newaxis] * (samples - mean)**2, axis=0)
   std = np.sqrt(var)
   
   # Print the parameter estimates
   param_names = ['t0', 'x0', 'x1', 'c'] if fixed_z is not None else ['z', 't0', 'x0', 'x1', 'c']
   print("\nParameter estimates:")
   for i, name in enumerate(param_names):
       print(f"{name} = {mean[i]:.6f} ± {std[i]:.6f}")

Visualizing the Results
--------------------

Finally, let's visualize the posterior distributions using corner plots:

.. code-block:: python

   import anesthetic
   import matplotlib.pyplot as plt
   
   # Create an anesthetic namespace
   ns = anesthetic.NestedSamples(
       data=result.samples,
       weights=result.weights,
       logL=result.logl,
       logL_birth=result.logls,
       labels=param_names
   )
   
   # Create a corner plot
   fig = ns.plot_2d(param_names)
   plt.tight_layout()
   plt.show()
   
   # Plot the posterior distributions
   fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params))
   for i, name in enumerate(param_names):
       ns.plot_1d(name, ax=axes[i])
       axes[i].set_xlabel(name)
       axes[i].set_ylabel('Probability density')
   plt.tight_layout()
   plt.show()

Plotting the Light Curves
----------------------

Let's also plot the observed and model light curves using the posterior mean parameters:

.. code-block:: python

   # Create a dictionary with the posterior mean parameters
   best_params = {}
   for i, name in enumerate(param_names):
       best_params[name] = mean[i]
   
   if fixed_z is not None:
       best_params['z'] = fixed_z[0]
   
   # Compute model fluxes for all observations
   model_fluxes = salt3_bandflux(times, bridges, best_params, zp=zps)
   
   # Create a figure
   plt.figure(figsize=(10, 6))
   
   # Plot the observed and model light curves
   unique_bands = np.unique(band_indices)
   colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bands)))
   
   for i, band_idx in enumerate(unique_bands):
       mask = band_indices == band_idx
       plt.errorbar(
           times[mask] - best_params['t0'],
           fluxes[mask],
           yerr=fluxerrs[mask],
           fmt='o',
           color=colors[i],
           label=f'Band {band_idx}'
       )
       plt.plot(
           times[mask] - best_params['t0'],
           model_fluxes[mask],
           '-',
           color=colors[i]
       )
   
   plt.xlabel('Phase (days)')
   plt.ylabel('Flux')
   plt.legend()
   plt.title('Observed and Model Light Curves')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

Complete Example
--------------

Here's a complete example that demonstrates how to use nested sampling with JAX-bandflux:

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np
   import matplotlib.pyplot as plt
   import nestle
   import anesthetic
   
   from jax_supernovae.data import load_and_process_data
   from jax_supernovae.salt3 import salt3_bandflux
   
   # Load and process data
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
       sn_name='19dwz',
       data_dir='data',
       fix_z=True
   )
   
   # Define the log-likelihood function
   def log_likelihood(parameters):
       # Create a dictionary containing parameters
       params = {
           'z': fixed_z[0] if fixed_z is not None else parameters[0],
           't0': parameters[0 if fixed_z is not None else 1],
           'x0': parameters[1 if fixed_z is not None else 2],
           'x1': parameters[2 if fixed_z is not None else 3],
           'c': parameters[3 if fixed_z is not None else 4]
       }
       
       # Compute model fluxes for all observations
       model_fluxes = salt3_bandflux(times, bridges, params, zp=zps)
       
       # Calculate the log-likelihood (assuming Gaussian errors)
       chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
       log_like = -0.5 * chi2
       
       return float(log_like)
   
   # Define the prior transform function
   def prior_transform(unit_cube):
       """Transform from the unit cube to the parameter space."""
       # Define the parameter ranges
       if fixed_z is not None:
           # t0, x0, x1, c
           ranges = [
               (-10, 10),       # t0
               (1e-10, 1e-2),   # x0
               (-3, 3),         # x1
               (-1, 1)          # c
           ]
       else:
           # z, t0, x0, x1, c
           ranges = [
               (0.01, 0.2),     # z
               (-10, 10),       # t0
               (1e-10, 1e-2),   # x0
               (-3, 3),         # x1
               (-1, 1)          # c
           ]
       
       # Transform from unit cube to parameter space
       params = np.zeros_like(unit_cube)
       for i, (lower, upper) in enumerate(ranges):
           if i == 2:  # x0 (log-uniform)
               params[i] = 10**(np.log10(lower) + unit_cube[i] * (np.log10(upper) - np.log10(lower)))
           else:  # uniform
               params[i] = lower + unit_cube[i] * (upper - lower)
       
       return params
   
   # Number of parameters
   n_params = 4 if fixed_z is not None else 5
   
   # Parameter names
   param_names = ['t0', 'x0', 'x1', 'c'] if fixed_z is not None else ['z', 't0', 'x0', 'x1', 'c']
   
   # Run nested sampling
   result = nestle.sample(
       log_likelihood,
       prior_transform,
       n_params,
       method='multi',
       npoints=1000,
       callback=lambda info: print(f"Iteration {info['it']}, log(Z) = {info['logz']:.2f}")
   )
   
   # Print the results
   print("Nested sampling completed!")
   print(f"Log evidence: {result.logz:.2f} ± {result.logzerr:.2f}")
   
   # Extract posterior samples
   weights = np.exp(result.logwt - result.logz)
   samples = result.samples
   
   # Compute weighted mean and standard deviation
   mean = np.sum(samples * weights[:, np.newaxis], axis=0)
   var = np.sum(weights[:, np.newaxis] * (samples - mean)**2, axis=0)
   std = np.sqrt(var)
   
   # Print the parameter estimates
   print("\nParameter estimates:")
   for i, name in enumerate(param_names):
       print(f"{name} = {mean[i]:.6f} ± {std[i]:.6f}")
   
   # Create an anesthetic namespace
   ns = anesthetic.NestedSamples(
       data=result.samples,
       weights=result.weights,
       logL=result.logl,
       logL_birth=result.logls,
       labels=param_names
   )
   
   # Create a corner plot
   fig = ns.plot_2d(param_names)
   plt.tight_layout()
   plt.show()
   
   # Plot the posterior distributions
   fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params))
   for i, name in enumerate(param_names):
       ns.plot_1d(name, ax=axes[i])
       axes[i].set_xlabel(name)
       axes[i].set_ylabel('Probability density')
   plt.tight_layout()
   plt.show()
   
   # Create a dictionary with the posterior mean parameters
   best_params = {}
   for i, name in enumerate(param_names):
       best_params[name] = mean[i]
   
   if fixed_z is not None:
       best_params['z'] = fixed_z[0]
   
   # Compute model fluxes for all observations
   model_fluxes = salt3_bandflux(times, bridges, best_params, zp=zps)
   
   # Create a figure
   plt.figure(figsize=(10, 6))
   
   # Plot the observed and model light curves
   unique_bands = np.unique(band_indices)
   colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bands)))
   
   for i, band_idx in enumerate(unique_bands):
       mask = band_indices == band_idx
       plt.errorbar(
           times[mask] - best_params['t0'],
           fluxes[mask],
           yerr=fluxerrs[mask],
           fmt='o',
           color=colors[i],
           label=f'Band {band_idx}'
       )
       plt.plot(
           times[mask] - best_params['t0'],
           model_fluxes[mask],
           '-',
           color=colors[i]
       )
   
   plt.xlabel('Phase (days)')
   plt.ylabel('Flux')
   plt.legend()
   plt.title('Observed and Model Light Curves')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()