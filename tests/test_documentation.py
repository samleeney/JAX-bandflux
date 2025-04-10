import os
import sys
import pytest
import jax
import jax.numpy as jnp
from jax_supernovae.data import load_and_process_data
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.bandpasses import Bandpass, register_bandpass, get_bandpass
import numpy as np
from scipy.optimize import minimize

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

def test_quickstart():
    """Test code blocks from quickstart.rst.
    
    This test verifies that the code examples in the quickstart documentation
    run correctly by executing the key steps: loading data, calculating model
    fluxes, and computing chi-squared.
    """
    # Load data for SN 19dwz
    times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz')
    
    # Check if fixed_z is None and provide a default value if needed
    if fixed_z is None:
        z = 0.1
    else:
        z = fixed_z[0]
        
    # Define SALT3 parameters
    params = {'z': z, 't0': 58650.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
    
    # Calculate model fluxes
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
    # Index the model fluxes with band_indices to match observations
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    
    # Calculate chi-squared
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
    
    # Verify the test ran successfully
    assert len(model_fluxes) == len(times), "Model fluxes length doesn't match times length"
    assert chi2 > 0, "Chi-squared should be positive"

def test_data_loading():
    """Test code blocks from data_loading.rst.
    
    This test verifies that the data loading examples in the documentation
    work correctly, including both synthetic data generation and loading
    real supernova data.
    """
    # Test synthetic data generation
    times = jnp.linspace(58650, 58700, 20)
    band_names = ['sdss::g', 'sdss::r', 'sdss::i']
    
    # Create synthetic data
    all_times = []
    all_bands = []
    
    for band in band_names:
        all_times.extend(times)
        all_bands.extend([band] * len(times))
    
    # Verify synthetic data
    assert len(all_times) == len(times) * len(band_names)
    assert len(all_bands) == len(times) * len(band_names)
    
    # Test loading real data
    times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz')
    # Verify real data
    assert len(times) > 0, "No data loaded for SN 19dwz"
    # fixed_z might be None in some cases, so we'll just check that the data loading completed

def test_bandpass_loading():
    """Test code blocks from bandpass_loading.rst.
    
    This test verifies that the bandpass loading examples in the documentation
    work correctly, including creating, registering, and retrieving custom
    bandpass objects.
    """
    # Test creating a custom bandpass
    wavelengths = np.linspace(4000, 5000, 100)
    transmission = np.exp(-((wavelengths - 4500) / 200)**2)
    
    # Create bandpass without name parameter (corrected from documentation)
    bandpass = Bandpass(wavelengths, transmission)
    
    # Register the bandpass
    register_bandpass('custom::g', bandpass)
    
    # Retrieve the bandpass
    retrieved_bandpass = get_bandpass('custom::g')
    
    # Verify bandpass
    assert retrieved_bandpass is not None, "Failed to retrieve registered bandpass"
    # Check the correct attribute names for the Bandpass object
    assert hasattr(retrieved_bandpass, 'wave'), "Bandpass should have 'wave' attribute"
    assert hasattr(retrieved_bandpass, 'trans'), "Bandpass should have 'trans' attribute"
    assert np.array_equal(retrieved_bandpass.wave, wavelengths), "Wavelength arrays don't match"
    assert np.array_equal(retrieved_bandpass.trans, transmission), "Transmission arrays don't match"

def test_model_fluxes():
    """Test code blocks from model_fluxes.rst.
    
    This test verifies that the model flux calculation examples in the
    documentation work correctly, including loading data, calculating
    model fluxes, and computing chi-squared.
    """
    # Load data
    times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz')
    
    # Check if fixed_z is None and provide a default value if needed
    if fixed_z is None:
        z = 0.1
    else:
        z = fixed_z[0]
    
    # Define SALT3 parameters
    params = {'z': z, 't0': 58650.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
    
    # Calculate model fluxes
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    
    # Calculate chi-squared
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
    
    # Verify model fluxes
    assert len(model_fluxes) == len(times), "Model fluxes length doesn't match times length"
    assert chi2 > 0, "Chi-squared should be positive"

def test_sampling():
    """Test code blocks from sampling.rst.
    
    This test verifies that the sampling examples in the documentation
    work correctly, including setting up an objective function and
    running a simple optimization.
    """
    # Load data
    times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz')
    
    # Check if fixed_z is None and provide a default value if needed
    if fixed_z is None:
        z = 0.1
    else:
        z = fixed_z[0]
    
    # Define objective function
    def objective(x):
        params = {
            'z': z,
            't0': x[0],
            'x0': x[1],
            'x1': x[2],
            'c': x[3]
        }
        
        model_fluxes = optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
        model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
        
        chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
        return chi2
    
    # Initial guess
    x0 = [58650.0, 1e-5, 0.0, 0.0]
    
    # Run a quick optimization (reduced iterations for testing)
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10})
    
    # Verify optimization
    assert result.fun > 0, "Objective function value should be positive"
    assert hasattr(result, 'x'), "Optimization result should have parameters"