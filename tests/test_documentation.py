"""Tests that verify documentation code examples work with SALT3Source v3.0 API.

This test validates that the code examples in the documentation run correctly
with the v3.0 functional API where parameters are passed as dictionaries to
bandflux() method instead of being stored in the source object.
"""

import os
import sys
import pytest
import jax
import jax.numpy as jnp
from jax_supernovae import SALT3Source, load_and_process_data
from jax_supernovae.bandpasses import Bandpass, register_bandpass, get_bandpass
import numpy as np
from scipy.optimize import minimize

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Enable float64 precision
jax.config.update("jax_enable_x64", True)


def test_quickstart():
    """Test code blocks from quickstart.rst using v3.0 functional API.

    This test verifies that the quickstart examples work correctly by executing
    the key steps: loading data, setting up params dict, and calculating
    model fluxes with v3.0 functional API.
    """
    # Load data for SN 19dwz
    times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = load_and_process_data('19dwz', fix_z=True)

    # Create source separately with v3.0 API
    source = SALT3Source()

    # Extract data
    z = fixed_z[0] if fixed_z else 0.1

    # Create params dict (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': 0.0,
        'c': 0.0
    }

    # Calculate phase in rest frame
    t0 = 58650.0
    phase_array = (times - t0) / (1 + z)

    # Calculate model fluxes using v3.0 functional API with optimized path
    model_fluxes = source.bandflux(params, None, phase_array, zp=zps, zpsys='ab',
                                    band_indices=band_indices, bridges=bridges,
                                    unique_bands=unique_bands)

    # Calculate chi-squared
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)

    # Verify the test ran successfully
    assert len(model_fluxes) == len(times), "Model fluxes length doesn't match times length"
    assert chi2 > 0, "Chi-squared should be positive"


def test_data_loading():
    """Test code blocks from data_loading.rst using new API.

    This test verifies that the data loading examples in the documentation
    work correctly with the new API.
    """
    # Test synthetic data generation
    times_synth = jnp.linspace(58650, 58700, 20)
    band_names = ['g', 'r', 'i']

    # Create synthetic data
    all_times = []
    all_bands = []

    for band in band_names:
        all_times.extend(times_synth)
        all_bands.extend([band] * len(times_synth))

    # Verify synthetic data
    assert len(all_times) == len(times_synth) * len(band_names)
    assert len(all_bands) == len(times_synth) * len(band_names)

    # Test loading real data with new API
    times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = load_and_process_data('19dwz', fix_z=True)

    # Verify real data structure
    assert times is not None, "Times should be loaded"
    assert unique_bands is not None, "Bands should be loaded"
    assert fluxes is not None, "Fluxes should be loaded"
    assert len(times) > 0, "No data loaded for SN 19dwz"


def test_bandpass_loading():
    """Test code blocks from bandpass_loading.rst.

    This test verifies that the bandpass loading examples in the documentation
    work correctly, including creating, registering, and retrieving custom
    bandpass objects.
    """
    # Test creating a custom bandpass
    wavelengths = np.linspace(4000, 5000, 100)
    transmission = np.exp(-((wavelengths - 4500) / 200)**2)

    # Create bandpass
    bandpass = Bandpass(wavelengths, transmission, name='custom::g')

    # Register the bandpass
    register_bandpass('custom::g', bandpass)

    # Retrieve the bandpass
    retrieved_bandpass = get_bandpass('custom::g')

    # Verify bandpass
    assert retrieved_bandpass is not None, "Failed to retrieve registered bandpass"
    assert hasattr(retrieved_bandpass, 'wave'), "Bandpass should have 'wave' attribute"
    assert hasattr(retrieved_bandpass, 'trans'), "Bandpass should have 'trans' attribute"
    assert retrieved_bandpass.name == 'custom::g', "Bandpass name should be preserved"
    assert np.array_equal(retrieved_bandpass.wave, wavelengths), "Wavelength arrays don't match"
    assert np.array_equal(retrieved_bandpass.trans, transmission), "Transmission arrays don't match"


def test_model_fluxes():
    """Test code blocks from model_fluxes.rst using v3.0 functional API.

    This test verifies that the model flux calculation examples in the
    documentation work correctly with the v3.0 functional API.
    """
    # Load data
    times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = load_and_process_data('19dwz', fix_z=True)

    # Create source separately with v3.0 API
    source = SALT3Source()

    # Extract data
    z = fixed_z[0] if fixed_z else 0.1

    # Create params dict (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': 0.0,
        'c': 0.0
    }

    # Calculate phase
    t0 = 58650.0
    phase_array = (times - t0) / (1 + z)

    # Calculate model fluxes using v3.0 functional API with optimized path
    model_fluxes = source.bandflux(params, None, phase_array, zp=zps, zpsys='ab',
                                    band_indices=band_indices, bridges=bridges,
                                    unique_bands=unique_bands)

    # Calculate chi-squared
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)

    # Verify model fluxes
    assert len(model_fluxes) == len(times), "Model fluxes length doesn't match times length"
    assert chi2 > 0, "Chi-squared should be positive"


def test_source_class_basic_usage():
    """Test basic usage of SALT3Source class with v3.0 functional API.

    This test demonstrates the v3.0 functional API and verifies it works correctly.
    """
    # Create a SALT3Source instance (v3.0 - no parameter storage)
    source = SALT3Source()

    # Create params dict (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': 0.5,
        'c': -0.1
    }

    # Test phase/wavelength bounds
    assert source.minphase() < 0
    assert source.maxphase() > 0
    assert source.minwave() > 1000
    assert source.maxwave() < 25000

    # Calculate flux for a single band at peak (v3.0 functional API)
    flux = source.bandflux(params, 'g', 0.0, zp=27.5, zpsys='ab')
    assert flux > 0, "Flux should be positive"


def test_source_class_arrays():
    """Test SALT3Source with arrays of bands and phases using v3.0 functional API."""
    # Create source (v3.0 - no parameter storage)
    source = SALT3Source()

    # Create params dict (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': 0.0,
        'c': 0.0
    }

    # Test with arrays
    bands = np.array(['g', 'r', 'i', 'z'])
    phases = np.array([0.0, 1.0, 2.0, 3.0])
    zps = np.array([27.5, 27.5, 27.5, 27.5])

    # Calculate fluxes (v3.0 functional API)
    fluxes = source.bandflux(params, bands, phases, zp=zps, zpsys='ab')

    # Verify results
    assert len(fluxes) == len(bands)
    assert len(fluxes) == len(phases)
    assert np.all(fluxes > 0), "All fluxes should be positive"


def test_sampling():
    """Test code blocks from sampling.rst using v3.0 functional API.

    This test verifies that the sampling examples in the documentation
    work correctly with the v3.0 functional API.
    """
    # Load data
    times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = load_and_process_data('19dwz', fix_z=True)

    # Create source separately with v3.0 API
    source = SALT3Source()

    # Extract data
    z = fixed_z[0] if fixed_z else 0.1

    # Define objective function using v3.0 functional API
    def objective(x):
        t0, log_x0, x1, c = x
        x0 = 10 ** log_x0

        # Create params dict (v3.0 functional API)
        params = {
            'x0': x0,
            'x1': x1,
            'c': c
        }

        # Calculate phase
        phase_array = (times - t0) / (1 + z)

        # Calculate model fluxes using v3.0 functional API with optimized path
        model_fluxes = source.bandflux(params, None, phase_array, zp=zps, zpsys='ab',
                                        band_indices=band_indices, bridges=bridges,
                                        unique_bands=unique_bands)

        # Calculate chi2
        chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
        return float(chi2)

    # Initial guess
    x0 = [58650.0, -5.0, 0.0, 0.0]

    # Run a quick optimization (reduced iterations for testing)
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10})

    # Verify optimization
    assert result.fun > 0, "Objective function value should be positive"
    assert hasattr(result, 'x'), "Optimization result should have parameters"


if __name__ == "__main__":
    test_quickstart()
    test_data_loading()
    test_bandpass_loading()
    test_model_fluxes()
    test_source_class_basic_usage()
    test_source_class_arrays()
    test_sampling()
    print("All documentation tests passed!")
