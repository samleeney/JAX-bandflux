import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import sncosmo
import matplotlib.pyplot as plt
from jax_supernovae.salt3nir import salt3nir_bandflux
from jax_supernovae.bandpasses import Bandpass

def test_salt3nir_fitting_consistency():
    """Test that fitting with JAX implementation gives same results as sncosmo."""
    
    # Create synthetic test data
    z = 0.1  # redshift
    t0 = 0.0  # time of peak brightness
    x0 = 1e-5  # overall flux normalization
    x1 = 0.1   # shape parameter
    c = 0.2    # color parameter
    
    # Generate time points
    times = np.linspace(-10, 30, 20)
    
    # Set up sncosmo model for generating test data
    snc_model = sncosmo.Model(source='salt3-nir')
    snc_model.set(z=z, t0=t0, x0=x0, x1=x1, c=c)
    
    # Create test bandpasses
    band_names = ['bessellb', 'bessellv', 'bessellr']
    bands = [sncosmo.get_bandpass(name) for name in band_names]
    
    # Create synthetic observations
    data = {'time': [], 'band': [], 'flux': [], 'fluxerr': [], 
            'zp': [], 'zpsys': []}
    
    np.random.seed(42)  # for reproducibility
    for band_name, band in zip(band_names, bands):
        # Sample each band at all times
        for t in times:
            flux = snc_model.bandflux(band, t, zp=25.0, zpsys='ab')
            # Add noise
            fluxerr = 0.1 * flux
            flux += np.random.normal(0, fluxerr)
            
            data['time'].append(t)
            data['band'].append(band_name)
            data['flux'].append(flux)
            data['fluxerr'].append(fluxerr)
            data['zp'].append(25.0)
            data['zpsys'].append('ab')
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    # Create JAX bandpasses
    band_dict = {}
    for band_name in band_names:
        snc_band = sncosmo.get_bandpass(band_name)
        band_dict[band_name] = Bandpass(snc_band.wave, snc_band.trans)
    
    # Define objective function for sncosmo
    def objective_sncosmo(parameters):
        snc_model.parameters[:] = parameters
        model_flux = snc_model.bandflux(data['band'], data['time'],
                                      zp=data['zp'], zpsys=data['zpsys'])
        return np.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)
    
    # Define objective function for JAX implementation
    def objective_jax(parameters):
        params = {
            'z': parameters[0],
            't0': parameters[1],
            'x0': parameters[2],
            'x1': parameters[3],
            'c': parameters[4]
        }
        
        # Calculate model flux for each observation
        model_flux = np.zeros_like(data['flux'])
        for i, (band_name, t, zp, zpsys) in enumerate(zip(data['band'], data['time'], 
                                                         data['zp'], data['zpsys'])):
            model_flux[i] = salt3nir_bandflux(t, band_dict[band_name], params, 
                                            zp=zp, zpsys=zpsys)
        
        return np.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)
    
    # Starting parameters and bounds (same for both)
    start_parameters = [0.05, 5.0, 5e-6, 0.5, 0.1]  # z, t0, x0, x1, c
    bounds = [(0.0, 0.3), (-10., 10.), (None, None), (-3., 3.), (-0.3, 0.3)]
    
    print("\nFitting with sncosmo...")
    snc_parameters, snc_val, snc_info = fmin_l_bfgs_b(
        objective_sncosmo, 
        start_parameters,
        bounds=bounds, 
        approx_grad=True
    )
    
    print("\nFitting with JAX implementation...")
    jax_parameters, jax_val, jax_info = fmin_l_bfgs_b(
        objective_jax, 
        start_parameters,
        bounds=bounds, 
        approx_grad=True
    )
    
    # Print results
    param_names = ['z', 't0', 'x0', 'x1', 'c']
    print("\nTrue Values:")
    true_values = [z, t0, x0, x1, c]
    for name, val in zip(param_names, true_values):
        print(f"{name:>10}: {val:10.6f}")
    
    print("\nFitting Results Comparison:")
    print("-" * 60)
    print(f"{'Parameter':>10} {'SNCosmo':>15} {'JAX':>15} {'Ratio':>10}")
    print("-" * 60)
    
    for name, snc_val, jax_val in zip(param_names, snc_parameters, jax_parameters):
        ratio = jax_val/snc_val if snc_val != 0 else 1.0
        print(f"{name:>10} {snc_val:15.6e} {jax_val:15.6e} {ratio:10.4f}")
    
    print("-" * 60)
    print(f"{'Chi2':>10} {snc_val:15.6e} {jax_val:15.6e}")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Plot data points
    for band_name in band_names:
        mask = data['band'] == band_name
        plt.errorbar(data['time'][mask], data['flux'][mask],
                    yerr=data['fluxerr'][mask],
                    fmt='o', alpha=0.5, label=f'Data {band_name}')
    
    # Plot best-fit models
    plot_times = np.linspace(times.min(), times.max(), 100)
    
    # SNCosmo model curves
    snc_model.parameters = snc_parameters
    for band_name in band_names:
        model_flux = snc_model.bandflux(band_name, plot_times,
                                      zp=25.0, zpsys='ab')
        plt.plot(plot_times, model_flux, '-', label=f'SNCosmo {band_name}')
    
    # JAX model curves
    jax_params = dict(zip(param_names, jax_parameters))
    for band_name in band_names:
        model_flux = salt3nir_bandflux(plot_times, band_dict[band_name], jax_params,
                                     zp=25.0, zpsys='ab')
        plt.plot(plot_times, model_flux, '--', label=f'JAX {band_name}')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Flux')
    plt.title('SN Light Curve - Data and Best Fit Models')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.savefig('salt3nir_fitting_comparison.png')
    
    # Assert that parameters match within tolerance
    np.testing.assert_allclose(jax_parameters, snc_parameters, rtol=1e-2,
                             err_msg="Best-fit parameters do not match")
    np.testing.assert_allclose(jax_val, snc_val, rtol=1e-2,
                             err_msg="Best-fit chi-square values do not match")

if __name__ == "__main__":
    test_salt3nir_fitting_consistency()
    print("\nAll tests passed.") 