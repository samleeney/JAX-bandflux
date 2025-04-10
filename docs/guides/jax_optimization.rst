JAX Optimization
==============

This guide provides in-depth information about JAX optimization techniques in JAX-bandflux.

Introduction to JAX
----------------

JAX is a high-performance numerical computing library that combines NumPy's familiar API with the power of automatic differentiation and GPU/TPU acceleration. JAX-bandflux leverages JAX to implement differentiable supernova light curve modeling, enabling efficient gradient-based optimization and large-scale computations.

Key features of JAX used in JAX-bandflux include:

1. **Automatic Differentiation**: JAX can automatically compute derivatives of functions, which is useful for gradient-based optimization.
2. **Just-in-Time (JIT) Compilation**: JAX can compile functions for faster execution.
3. **Vectorization**: JAX can vectorize functions for parallel computation.
4. **GPU/TPU Acceleration**: JAX can run computations on GPUs and TPUs for faster execution.

Automatic Differentiation
----------------------

JAX provides automatic differentiation through the ``jax.grad`` function. This allows you to compute derivatives of functions with respect to their inputs:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax_supernovae.salt3 import salt3_bandflux
   
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

You can also compute higher-order derivatives using ``jax.hessian`` or ``jax.jacfwd`` and ``jax.jacrev``:

.. code-block:: python

   # Compute the Hessian
   hessian = jax.hessian(compute_bandflux)(params_array)
   
   print("Hessian shape:", hessian.shape)

Just-in-Time (JIT) Compilation
---------------------------

JAX provides just-in-time (JIT) compilation through the ``jax.jit`` function. This can significantly improve performance for repeated evaluations of the same function:

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

JIT compilation works best for functions that don't change their computation graph between calls. For example, if you have a function that uses different control flow based on its inputs, JIT compilation may not be as effective.

Vectorization
-----------

JAX provides vectorization through the ``jax.vmap`` function. This allows you to apply a function to multiple inputs in parallel:

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

You can also vectorize over multiple inputs:

.. code-block:: python

   # Define a function that computes the bandflux for a single time and bridge
   def compute_bandflux_single(time, bridge, params):
       return salt3_bandflux(time, bridge, params)
   
   # Vectorize the function over times and bridges
   compute_bandflux_vectorized = jax.vmap(compute_bandflux_single, in_axes=(0, 0, None))
   
   # Compute bandfluxes for multiple times and bridges
   times = jnp.linspace(-10, 30, 100)
   bridges = [bridge] * 100  # Just an example, you would use actual bridges
   fluxes = compute_bandflux_vectorized(times, bridges, params)
   
   print("Fluxes shape:", fluxes.shape)

Vectorization can significantly improve performance for large-scale computations.

GPU/TPU Acceleration
-----------------

JAX can run computations on GPUs and TPUs for faster execution. To use GPU acceleration, you need to install the GPU version of JAX. Please refer to the `JAX installation guide <https://github.com/google/jax#installation>`_ for detailed instructions.

Once you have installed the GPU version of JAX, you can use JAX-bandflux as usual, and JAX will automatically use the GPU for computations.

You can check if JAX is using the GPU:

.. code-block:: python

   import jax
   
   print("Devices:", jax.devices())

If you see a GPU device in the list, JAX is using the GPU.

Optimization Algorithms
--------------------

JAX-bandflux supports various optimization algorithms for fitting SALT parameters to supernova light curve data:

1. **L-BFGS-B**: A gradient-based optimization algorithm that approximates the Hessian matrix.
2. **Nested Sampling**: A Bayesian inference algorithm that explores the parameter space and computes the Bayesian evidence.

L-BFGS-B is implemented using SciPy's ``minimize`` function:

.. code-block:: python

   from scipy.optimize import minimize
   import numpy as np
   
   # Define the objective function
   def objective(parameters):
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
       
       # Calculate the chi-squared statistic
       chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
       
       return float(chi2)
   
   # Initial parameter values
   if fixed_z is not None:
       initial_params = np.array([0.0, 1e-5, 0.0, 0.0])  # t0, x0, x1, c
   else:
       initial_params = np.array([0.1, 0.0, 1e-5, 0.0, 0.0])  # z, t0, x0, x1, c
   
   # Parameter bounds
   if fixed_z is not None:
       bounds = [(-10, 10), (1e-10, 1e-2), (-3, 3), (-1, 1)]  # t0, x0, x1, c
   else:
       bounds = [(0.01, 0.2), (-10, 10), (1e-10, 1e-2), (-3, 3), (-1, 1)]  # z, t0, x0, x1, c
   
   # Optimize the parameters
   result = minimize(
       objective,
       initial_params,
       method='L-BFGS-B',
       bounds=bounds,
       options={'disp': True}
   )
   
   # Print the results
   print("Optimization successful:", result.success)
   print("Number of function evaluations:", result.nfev)
   
   # Extract the best-fit parameters
   if fixed_z is not None:
       best_t0, best_x0, best_x1, best_c = result.x
       best_z = fixed_z[0]
   else:
       best_z, best_t0, best_x0, best_x1, best_c = result.x
   
   print("Best-fit parameters:")
   print(f"z = {best_z:.6f}")
   print(f"t0 = {best_t0:.6f}")
   print(f"x0 = {best_x0:.6e}")
   print(f"x1 = {best_x1:.6f}")
   print(f"c = {best_c:.6f}")

Nested sampling is implemented using the ``nestle`` package:

.. code-block:: python

   import nestle
   
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

Performance Optimization
---------------------

Here are some tips for optimizing the performance of JAX-bandflux:

1. **Use JIT Compilation**: JIT-compile functions that are called repeatedly.
2. **Use Vectorization**: Vectorize functions that are applied to multiple inputs.
3. **Use GPU Acceleration**: Use the GPU version of JAX for faster execution.
4. **Precompute Bridge Data**: Precompute bridge data for each bandpass to avoid recomputing it for each flux calculation.
5. **Use Efficient Data Structures**: Use JAX arrays instead of NumPy arrays for efficient computation.
6. **Minimize Host-Device Transfers**: Minimize transfers between the CPU and GPU by keeping data on the GPU as much as possible.
7. **Use Appropriate Batch Sizes**: Use appropriate batch sizes for vectorized computations to balance memory usage and performance.

By following these tips, you can significantly improve the performance of JAX-bandflux for large-scale computations.