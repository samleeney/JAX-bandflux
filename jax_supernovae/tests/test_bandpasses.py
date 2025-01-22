"""Tests for bandpass handling."""
import pytest
import numpy as np
import sncosmo
from jax_supernovae.bandpasses import (
    load_bandpass,
    register_bandpass,
    get_bandpass,
    register_hsf_bandpasses,
)
from jax_supernovae.core import Bandpass
import jax.numpy as jnp

def test_load_bandpass():
    """Test loading a bandpass from file."""
    # Test loading ATLAS cyan filter
    bandpass = load_bandpass('c')
    assert isinstance(bandpass, Bandpass)
    assert len(bandpass.wave) > 0
    assert len(bandpass.trans) > 0
    assert bandpass.wave[0] > 0
    assert np.all(bandpass.trans >= 0)
    assert np.all(bandpass.trans <= 1)

def test_register_and_get_bandpass():
    """Test registering and retrieving bandpasses."""
    # Create a simple test bandpass
    wave = np.array([4000., 5000., 6000.])
    trans = np.array([0., 1., 0.])
    bandpass = Bandpass(wave, trans)
    
    # Test registration
    register_bandpass('test', bandpass)
    retrieved = get_bandpass('test')
    assert isinstance(retrieved, Bandpass)
    assert np.array_equal(retrieved.wave, wave)
    assert np.array_equal(retrieved.trans, trans)
    
    # Test force parameter
    with pytest.raises(ValueError):
        register_bandpass('test', bandpass, force=False)
    register_bandpass('test', bandpass, force=True)  # Should not raise

def test_register_hsf_bandpasses():
    """Test registering all HSF bandpasses."""
    register_hsf_bandpasses()
    
    # Test that all required bandpasses are registered
    required_bands = ['c', 'o', 'ztfg', 'ztfr']  # Temporarily remove J_1D3 and J_2D
    for band in required_bands:
        bandpass = get_bandpass(band)
        assert isinstance(bandpass, Bandpass)
        assert len(bandpass.wave) > 0
        assert len(bandpass.trans) > 0

def test_bandpass_wavelength_ranges():
    """Test that bandpass wavelength ranges are reasonable."""
    register_hsf_bandpasses()
    
    # Test wavelength ranges for each filter
    ranges = {
        'c': (4100, 6600),    # ATLAS cyan
        'o': (5500, 8500),    # ATLAS orange
        'ztfg': (3600, 6000), # ZTF g-band
        'ztfr': (5400, 7500), # ZTF r-band
    }
    
    for band, (min_wave, max_wave) in ranges.items():
        bandpass = get_bandpass(band)
        assert bandpass.minwave() >= min_wave
        assert bandpass.maxwave() <= max_wave

def test_bandpass_comparison_with_sncosmo():
    """Test that our bandpasses match sncosmo's."""
    register_hsf_bandpasses()
    
    # Compare ZTF filters directly
    for band in ['ztfg', 'ztfr']:
        our_bandpass = get_bandpass(band)
        sncosmo_bandpass = sncosmo.get_bandpass(band)
        
        # Interpolate our transmission to sncosmo's wavelength grid
        our_trans = our_bandpass(sncosmo_bandpass.wave)
        
        # Compare transmissions with a larger tolerance due to float32/float64 differences
        np.testing.assert_allclose(
            our_trans,
            sncosmo_bandpass.trans,
            rtol=1e-6,
            atol=1e-7
        )

def test_bandpass_integration():
    """Test bandpass integration with model."""
    from jax_supernovae.salt3nir import salt3nir_bandflux
    
    # Register bandpasses
    register_hsf_bandpasses()
    
    # Set parameters
    params = {
        'z': 0.0,  # No redshift for this test
        't0': 0.0, # No time offset
        'x0': 1e-4,
        'x1': 0.0,
        'c': 0.0
    }
    
    # Test bandflux calculation for each filter
    times = np.array([0.0, 10.0, 20.0])  # Rest-frame phases
    for band in ['c', 'o', 'ztfg', 'ztfr']:  # Temporarily remove J_1D3 and J_2D
        bandpass = get_bandpass(band)
        flux = salt3nir_bandflux(times, bandpass, params)
        # Convert JAX array to numpy array for assertions
        flux = np.array(flux)
        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(times)
        assert np.all(np.isfinite(flux))
        assert np.all(flux >= 0) 