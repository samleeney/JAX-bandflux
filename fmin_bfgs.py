import jax
import jax.numpy as jnp
from scipy.optimize import fmin_l_bfgs_b
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.data import load_and_process_data
import numpy as np

# Enable float64 precision for better accuracy
jax.config.update("jax_enable_x64", True)

def fit_salt3(fix_z=True, sn_name="19dwz"):
    """
    Fit SALT3 model to supernova data using L-BFGS-B optimisation.
    
    Parameters
    ----------
    fix_z : bool, optional
        Whether to fix the redshift to a known value (default: True)
    sn_name : str, optional
        Name of supernova for redshift lookup (default: "19dwz")
    """
    # Load and process data using the utility function
    times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
        sn_name, data_dir='data', fix_z=fix_z
    )

    def objective(parameters):
        """Calculate chi-squared for given parameters."""
        try:
            if fix_z:
                param_dict = {
                    'z': fixed_z[0],
                    't0': parameters[0],
                    'x0': parameters[1],
                    'x1': parameters[2],
                    'c': parameters[3]
                }
            else:
                param_dict = {
                    'z': parameters[0],
                    't0': parameters[1],
                    'x0': parameters[2],
                    'x1': parameters[3],
                    'c': parameters[4]
                }
            
            # Calculate model fluxes
            model_fluxes = optimized_salt3_multiband_flux(
                times, bridges, param_dict, zps=zps, zpsys='ab'
            )
            model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
            
            # Calculate chi-squared
            chi2 = float(jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2))
            
            # Print progress
            param_names = ['t0', 'x0', 'x1', 'c'] if fix_z else ['z', 't0', 'x0', 'x1', 'c']
            print("\nCurrent parameters:")
            for name, value in zip(param_names, parameters):
                print(f"{name:>10} = {value:.6f}")
            print(f"Chi2: {chi2:.2f}")
            
            return chi2
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e12

    # Starting parameter values
    if fix_z:
        start_parameters = [
            58600.,     # t0
            1.26e-3,    # x0
            1.5,        # x1
            0.1         # c
        ]
        bounds = [
            (58500., 58700.),  # t0
            (1e-4, 1e-2),      # x0
            (-2.0, 2.0),       # x1
            (-0.2, 0.2)        # c
        ]
    else:
        start_parameters = [
            0.15,       # z
            58600.,     # t0
            1.26e-3,    # x0
            1.5,        # x1
            0.1         # c
        ]
        bounds = [
            (0.01, 0.3),       # z
            (58500., 58700.),  # t0
            (1e-4, 1e-2),      # x0
            (-2.0, 2.0),       # x1
            (-0.2, 0.2)        # c
        ]

    print("\nStarting optimisation...")
    print("Initial parameters:", start_parameters)

    # Run the optimisation
    parameters, val, info = fmin_l_bfgs_b(objective, start_parameters,
                                         bounds=bounds,
                                         approx_grad=True,
                                         maxfun=2000,
                                         maxiter=400)

    # Print results
    param_names = ['t0', 'x0', 'x1', 'c'] if fix_z else ['z', 't0', 'x0', 'x1', 'c']
    print("\nBest-fit parameters:")
    for name, value in zip(param_names, parameters):
        print(f"{name:>10} = {value:.6f}")
    if fix_z:
        print(f"{'z':>10} = {fixed_z[0]:.6f} (fixed)")
    print(f"\nFinal chi-squared: {val:.2f}")
    
    return parameters, val, info

if __name__ == "__main__":
    # Example usage with fixed redshift (default)
    parameters, val, info = fit_salt3()
    
    # For free redshift:
    # parameters, val, info = fit_salt3(fix_z=False) 