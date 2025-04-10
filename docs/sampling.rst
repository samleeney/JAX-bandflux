Sampling
=======

This section describes how to perform basic parameter sampling and optimization with JAX-Supernovae. We focus on defining objective functions and using simple optimization techniques to fit SALT3 model parameters to supernova light curve data.

Defining an Objective Function
---------------------------

The first step in parameter fitting is to define an objective function that quantifies the goodness of fit between model predictions and observed data. For supernova light curve fitting, a common choice is the chi-squared statistic:

.. code-block:: python

    import jax.numpy as jnp
    from jax_supernovae.salt3 import optimized_salt3_multiband_flux

    def objective(parameters):
        """
        Objective function for SALT3 parameter fitting.
        
        Parameters:
        - parameters: Array of [t0, x0, x1, c] (assuming fixed redshift)
        
        Returns:
        - chi2: Chi-squared value
        """
        # Create parameter dictionary
        params = {
            'z': fixed_z[0],  # Fixed redshift
            't0': parameters[0],
            'x0': parameters[1],
            'x1': parameters[2],
            'c': parameters[3]
        }
        
        # Calculate model fluxes
        model_fluxes = optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
        # Index the model fluxes with band_indices to match observations
        model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
        
        # Calculate chi-squared
        chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
        
        return chi2

Basic Sampling with scipy.optimize
-------------------------------

Once the objective function is defined, we can use optimization methods from SciPy to find the best-fit parameters:

.. code-block:: python

    from scipy.optimize import minimize
    import numpy as np

    # Initial parameter values
    initial_params = np.array([
        58650.0,  # t0
        1e-5,     # x0
        0.0,      # x1
        0.0       # c
    ])

    # Parameter bounds
    bounds = [
        (58600.0, 58700.0),  # t0
        (1e-6, 1e-4),        # x0
        (-3.0, 3.0),         # x1
        (-0.3, 0.3)          # c
    ]

    # Optimize the parameters
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds
    )

    # Print the results
    print("Optimization successful:", result.success)
    print("Number of function evaluations:", result.nfev)

    # Extract best-fit parameters
    best_params = {
        'z': fixed_z[0],
        't0': result.x[0],
        'x0': result.x[1],
        'x1': result.x[2],
        'c': result.x[3]
    }

    print("\nBest-fit parameters:")
    for name, value in best_params.items():
        print(f"{name:>10} = {value:.6f}")

    print(f"\nFinal chi-squared: {result.fun:.2f}")

Complete Sampling Example
----------------------

Here is a complete example that demonstrates the entire process of loading data, defining an objective function, and optimizing parameters:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize
    from jax_supernovae.data import load_and_process_data
    from jax_supernovae.salt3 import optimized_salt3_multiband_flux

    # Enable float64 precision
    jax.config.update("jax_enable_x64", True)

    # Load data
    times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
        sn_name='19dwz',
        data_dir='data',
        fix_z=True
    )

    # Define the objective function
    def objective(parameters):
        # Create parameter dictionary
        params = {
            'z': fixed_z[0],  # Fixed redshift
            't0': parameters[0],
            'x0': parameters[1],
            'x1': parameters[2],
            'c': parameters[3]
        }
        
        # Calculate model fluxes
        model_fluxes = optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
        # Index the model fluxes with band_indices to match observations
        model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
        
        # Calculate chi-squared
        chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
        
        return float(chi2)

    # Initial parameter values
    initial_params = np.array([
        58650.0,  # t0
        1e-5,     # x0
        0.0,      # x1
        0.0       # c
    ])

    # Parameter bounds
    bounds = [
        (58600.0, 58700.0),  # t0
        (1e-6, 1e-4),        # x0
        (-3.0, 3.0),         # x1
        (-0.3, 0.3)          # c
    ]

    # Optimize the parameters
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds
    )

    # Print the results
    print("Optimization successful:", result.success)
    print("Number of function evaluations:", result.nfev)

    # Extract best-fit parameters
    best_params = {
        'z': fixed_z[0],
        't0': result.x[0],
        'x0': result.x[1],
        'x1': result.x[2],
        'c': result.x[3]
    }

    print("\nBest-fit parameters:")
    for name, value in best_params.items():
        print(f"{name:>10} = {value:.6f}")

    print(f"\nFinal chi-squared: {result.fun:.2f}")