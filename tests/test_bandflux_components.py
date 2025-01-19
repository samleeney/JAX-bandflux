"""Tests for individual components of bandflux calculation."""
import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision

import jax.numpy as jnp
import numpy as np
import sncosmo
from jax_supernovae.core import Bandpass, HC_ERG_AA
from jax_supernovae.models import Model
from jax_supernovae.salt2 import salt2_flux, salt2_m0, salt2_m1, salt2_colorlaw
from jax_supernovae.salt2_data import get_salt2_wave_grid
from jax_supernovae.bandpasses import get_bandpass

def test_bandpass_components():
    """Test that bandpass components (wavelength and transmission) match between implementations."""
    # Initialize bandpass
    band_name = 'sdssg'
    sncosmo_band = sncosmo.get_bandpass(band_name)
    jax_band = get_bandpass(band_name)
    
    print("\nBandpass Components Test:")
    print("------------------------")
    
    # Compare wavelength grids
    print("\nWavelength grid comparison:")
    print(f"SNCosmo wave shape: {sncosmo_band.wave.shape}")
    print(f"JAX wave shape: {jax_band.wave.shape}")
    print(f"Max absolute difference in wavelengths: {np.max(np.abs(sncosmo_band.wave - jax_band.wave)):.2e}")
    assert np.allclose(sncosmo_band.wave, jax_band.wave), "Wavelength grids don't match"
    
    # Compare transmission values
    print("\nTransmission values comparison:")
    print(f"SNCosmo trans shape: {sncosmo_band.trans.shape}")
    print(f"JAX trans shape: {jax_band.trans.shape}")
    print(f"Max absolute difference in transmission: {np.max(np.abs(sncosmo_band.trans - jax_band.trans)):.2e}")
    assert np.allclose(sncosmo_band.trans, jax_band.trans), "Transmission values don't match"

def test_rest_frame_components():
    """Test that rest-frame calculations match between implementations."""
    # Initialize models with same parameters
    params = {'t0': 55000.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0, 'z': 0.5}
    sncosmo_model = sncosmo.Model(source='salt2')
    sncosmo_model.set(**params)
    
    jax_model = Model()
    jax_model.wave = get_salt2_wave_grid()
    jax_model._flux = lambda t, w: salt2_flux(t, w, jax_model.parameters)
    jax_model.parameters = params
    
    print("\nRest Frame Components Test:")
    print("-------------------------")
    
    # Test at a specific time and wavelength
    time = 55000.0
    wave = 4000.0
    
    # Calculate rest-frame values
    z = params['z']
    a = 1.0 / (1.0 + z)
    rest_phase = (time - params['t0']) * a
    rest_wave = wave * a
    
    print(f"\nTesting at:")
    print(f"Observer frame time: {time}")
    print(f"Observer frame wavelength: {wave}")
    print(f"Rest frame phase: {rest_phase}")
    print(f"Rest frame wavelength: {rest_wave}")
    
    # Get SNCosmo components
    sncosmo_m0 = sncosmo_model._source._model['M0'](np.array([rest_phase]), np.array([rest_wave]))[0][0]
    sncosmo_m1 = sncosmo_model._source._model['M1'](np.array([rest_phase]), np.array([rest_wave]))[0][0]
    sncosmo_cl = sncosmo_model._source._colorlaw(np.array([rest_wave]))[0]
    
    # Get JAX components
    jax_m0 = float(salt2_m0(rest_phase, rest_wave))
    jax_m1 = float(salt2_m1(rest_phase, rest_wave))
    jax_cl = float(salt2_colorlaw(rest_wave, jnp.array([-0.402687, 0.700296, -0.431342, 0.0779681])))
    
    print("\nM0 comparison:")
    print(f"SNCosmo M0: {sncosmo_m0:.6e}")
    print(f"JAX M0: {jax_m0:.6e}")
    print(f"Relative difference: {abs(sncosmo_m0 - jax_m0) / abs(sncosmo_m0):.2e}")
    assert np.allclose(sncosmo_m0, jax_m0), "M0 values don't match"
    
    print("\nM1 comparison:")
    print(f"SNCosmo M1: {sncosmo_m1:.6e}")
    print(f"JAX M1: {jax_m1:.6e}")
    print(f"Relative difference: {abs(sncosmo_m1 - jax_m1) / abs(sncosmo_m1):.2e}")
    assert np.allclose(sncosmo_m1, jax_m1), "M1 values don't match"
    
    print("\nColor law comparison:")
    print(f"SNCosmo CL: {sncosmo_cl:.6e}")
    print(f"JAX CL: {jax_cl:.6e}")
    print(f"Relative difference: {abs(sncosmo_cl - jax_cl) / abs(sncosmo_cl):.2e}")
    assert np.allclose(sncosmo_cl, jax_cl), "Color law values don't match"

def test_flux_integration():
    """Test that flux integration matches between implementations."""
    # Initialize models with same parameters
    params = {'t0': 55000.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0, 'z': 0.5}
    sncosmo_model = sncosmo.Model(source='salt2')
    sncosmo_model.set(**params)
    
    jax_model = Model()
    jax_model.wave = get_salt2_wave_grid()
    jax_model._flux = lambda t, w: salt2_flux(t, w, jax_model.parameters)
    jax_model.parameters = params
    
    print("\nFlux Integration Test:")
    print("--------------------")
    
    # Test at a specific time
    time = 55000.0
    band_name = 'sdssg'
    
    # Get bandfluxes
    sncosmo_flux = sncosmo_model.bandflux(band_name, time)
    jax_flux = jax_model.bandflux(band_name, time)
    
    print(f"\nTesting at time: {time}")
    print(f"SNCosmo bandflux: {sncosmo_flux:.6e}")
    print(f"JAX bandflux: {jax_flux:.6e}")
    print(f"Relative difference: {abs(sncosmo_flux - jax_flux) / abs(sncosmo_flux):.2e}")
    
    # Get the bandpass for detailed investigation
    band = get_bandpass(band_name)
    wave = band.wave
    trans = band.trans
    
    # Calculate rest-frame quantities
    z = params['z']
    a = 1.0 / (1.0 + z)
    rest_phase = (time - params['t0']) * a
    rest_wave = wave * a
    
    # Get fluxes at each wavelength
    sncosmo_fluxes = sncosmo_model._flux(rest_phase, rest_wave)
    jax_fluxes = jax_model._flux_with_redshift(np.array([time]), wave[None, :])[0]
    
    print("\nFlux comparison at wavelength points:")
    print(f"Max absolute difference in fluxes: {np.max(np.abs(sncosmo_fluxes - jax_fluxes)):.2e}")
    print(f"Mean absolute difference in fluxes: {np.mean(np.abs(sncosmo_fluxes - jax_fluxes)):.2e}")
    
    # Compare integration components
    dwave = wave[1] - wave[0]
    sncosmo_integrand = wave * trans * sncosmo_fluxes
    jax_integrand = wave * trans * jax_fluxes
    
    print("\nIntegration components comparison:")
    print(f"Max absolute difference in integrand: {np.max(np.abs(sncosmo_integrand - jax_integrand)):.2e}")
    print(f"Mean absolute difference in integrand: {np.mean(np.abs(sncosmo_integrand - jax_integrand)):.2e}")
    
    # Compare final integration results
    sncosmo_sum = np.sum(sncosmo_integrand) * dwave / HC_ERG_AA
    jax_sum = np.sum(jax_integrand) * dwave / HC_ERG_AA
    
    print("\nFinal integration results:")
    print(f"SNCosmo sum: {sncosmo_sum:.6e}")
    print(f"JAX sum: {jax_sum:.6e}")
    print(f"Relative difference: {abs(sncosmo_sum - jax_sum) / abs(sncosmo_sum):.2e}")
    
    assert np.allclose(sncosmo_flux, jax_flux, rtol=1e-5), "Bandflux values don't match"

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