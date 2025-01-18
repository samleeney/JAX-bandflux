"""Tests for individual components of bandflux calculation."""
import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision

import numpy as np
import sncosmo
from jax_supernovae.core import Bandpass, HC_ERG_AA
from jax_supernovae.models import Model
from jax_supernovae.salt2 import salt2_flux
from jax_supernovae.salt2_data import get_salt2_wave_grid

def test_bandpass_transmission():
    """Test that bandpass transmission matches between SNCosmo and JAX."""
    # Create a test bandpass
    wave = np.linspace(4000, 6000, 101)
    trans = np.exp(-(wave - 5000)**2 / 1000**2)
    
    # Create bandpass objects
    snc_band = sncosmo.Bandpass(wave, trans)
    jax_band = Bandpass(wave, trans)
    
    # Test transmission at original wavelengths
    jax_trans = jax_band(wave)
    np.testing.assert_allclose(jax_trans, trans, rtol=1e-5)
    
    # Test transmission at interpolated wavelengths
    wave_test = np.linspace(4500, 5500, 51)
    snc_trans = snc_band(wave_test)
    jax_trans = jax_band(wave_test)
    np.testing.assert_allclose(jax_trans, snc_trans, rtol=1e-5)

def test_flux_calculation():
    """Test that flux calculation matches between SNCosmo and JAX."""
    # Initialize models
    snc_model = sncosmo.Model(source='salt2')
    jax_model = Model()
    jax_model.wave = get_salt2_wave_grid()
    jax_model._flux = lambda t, w: salt2_flux(t, w, jax_model.parameters)
    
    # Set parameters
    params = {'t0': 55000.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0, 'z': 0.5}
    snc_model.set(**params)
    jax_model.parameters = params
    
    # Test flux at various times and wavelengths
    times = np.array([54990.0, 55000.0, 55010.0])
    waves = np.array([4000.0, 5000.0, 6000.0])
    
    for t in times:
        for w in waves:
            snc_flux = snc_model.flux(t, w)
            jax_flux = jax_model._flux_with_redshift(t, w)
            np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-5)

def test_integration_weights():
    """Test that integration weights match between SNCosmo and JAX."""
    # Create a test bandpass
    wave = np.linspace(4000, 6000, 101)
    trans = np.exp(-(wave - 5000)**2 / 1000**2)
    
    # Create bandpass objects
    snc_band = sncosmo.Bandpass(wave, trans)
    jax_band = Bandpass(wave, trans)
    
    # Calculate weights (wave * trans / HC_ERG_AA)
    snc_weights = wave * snc_band(wave) / HC_ERG_AA
    jax_weights = wave * jax_band(wave) / HC_ERG_AA
    
    np.testing.assert_allclose(jax_weights, snc_weights, rtol=1e-5)

def test_zeropoint_scaling():
    """Test that zeropoint scaling matches between SNCosmo and JAX."""
    # Initialize models
    snc_model = sncosmo.Model(source='salt2')
    jax_model = Model()
    jax_model.wave = get_salt2_wave_grid()
    jax_model._flux = lambda t, w: salt2_flux(t, w, jax_model.parameters)
    
    # Set parameters
    params = {'t0': 55000.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0, 'z': 0.5}
    snc_model.set(**params)
    jax_model.parameters = params
    
    # Test bandflux with and without zeropoint
    time = np.array([55000.0])
    band = 'sdssg'
    
    # Without zeropoint
    snc_flux = snc_model.bandflux(band, time)
    jax_flux = jax_model.bandflux(band, time)
    np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-5)
    
    # With zeropoint
    zp = 25.0
    zpsys = 'ab'
    snc_flux = snc_model.bandflux(band, time, zp=zp, zpsys=zpsys)
    jax_flux = jax_model.bandflux(band, time, zp=zp, zpsys=zpsys)
    np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-5) 