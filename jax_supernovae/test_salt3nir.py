import numpy as np
import pytest
import sncosmo
import os
from jax_supernovae.salt3nir import salt3nir_m0, salt3nir_m1, salt3nir_colorlaw, salt3nir_flux

# Ensure we're using the correct model file paths
os.environ['SNCOSMO_DATA_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sncosmo-modelfiles'))

# Test cases with known values
TEST_PHASES = [0.0, 10.0, 20.0]
TEST_WAVES = [4000.0, 5000.0, 6000.0]
TEST_PARAMS = {'x0': 1.0, 'x1': 0.1, 'c': 0.0}

def test_m0_matches_sncosmo():
    """Test that M0 component matches SNCosmo exactly."""
    snc_model = sncosmo.Model(source='salt3-nir')
    for phase in TEST_PHASES:
        for wave in TEST_WAVES:
            snc_m0 = snc_model._source._model['M0'](np.array([phase]), np.array([wave]))[0][0]
            jax_m0 = salt3nir_m0(phase, wave)
            np.testing.assert_allclose(
                jax_m0, snc_m0, rtol=1e-15,
                err_msg=f"M0 mismatch at phase={phase}, wave={wave}"
            )

def test_m1_matches_sncosmo():
    """Test that M1 component matches SNCosmo exactly."""
    snc_model = sncosmo.Model(source='salt3-nir')
    for phase in TEST_PHASES:
        for wave in TEST_WAVES:
            snc_m1 = snc_model._source._model['M1'](np.array([phase]), np.array([wave]))[0][0]
            jax_m1 = salt3nir_m1(phase, wave)
            np.testing.assert_allclose(
                jax_m1, snc_m1, rtol=1e-15,
                err_msg=f"M1 mismatch at phase={phase}, wave={wave}"
            )

def test_colorlaw_matches_sncosmo():
    """Test that color law matches SNCosmo exactly."""
    snc_model = sncosmo.Model(source='salt3-nir')
    for wave in TEST_WAVES:
        snc_cl = snc_model.source.colorlaw(np.array([wave]))
        jax_cl = salt3nir_colorlaw(wave)
        np.testing.assert_allclose(
            jax_cl, snc_cl, rtol=1e-15,
            err_msg=f"Color law mismatch at wave={wave}"
        )

def test_flux_calculation():
    """Test the full flux calculation with specific parameters."""
    params = {'x0': 1.0, 'x1': 0.1, 'c': 0.2}
    for phase in TEST_PHASES:
        for wave in TEST_WAVES:
            # Calculate components
            m0 = salt3nir_m0(phase, wave)
            m1 = salt3nir_m1(phase, wave)
            cl = salt3nir_colorlaw(wave)
            
            # Calculate expected flux manually
            expected_flux = params['x0'] * (m0 + params['x1'] * m1) * 10**(-0.4 * cl * params['c'])
            
            # Calculate flux using the function
            actual_flux = salt3nir_flux(phase, wave, params)
            
            np.testing.assert_allclose(
                actual_flux, expected_flux, rtol=1e-15,
                err_msg=f"Flux mismatch at phase={phase}, wave={wave}"
            )

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test at phase grid boundaries
    phase_min = -20.0  # Known minimum phase
    phase_max = 50.0   # Known maximum phase
    wave_mid = 5000.0
    
    # Test at wavelength grid boundaries
    wave_min = 2000.0  # Known minimum wavelength
    wave_max = 20000.0 # Known maximum wavelength
    phase_mid = 0.0
    
    # Test phase boundaries
    m0_phase_min = salt3nir_m0(phase_min, wave_mid)
    m0_phase_max = salt3nir_m0(phase_max, wave_mid)
    assert m0_phase_min is not None, "M0 should handle minimum phase"
    assert m0_phase_max is not None, "M0 should handle maximum phase"
    
    # Test wavelength boundaries
    m0_wave_min = salt3nir_m0(phase_mid, wave_min)
    m0_wave_max = salt3nir_m0(phase_mid, wave_max)
    assert m0_wave_min is not None, "M0 should handle minimum wavelength"
    assert m0_wave_max is not None, "M0 should handle maximum wavelength"

def test_vectorized_input():
    """Test that functions handle vectorized input correctly."""
    # Test multiple phases, single wavelength
    phases = np.array(TEST_PHASES)
    wave = TEST_WAVES[0]
    m0_vec = salt3nir_m0(phases, wave)
    assert len(m0_vec) == len(phases), "Vectorized phase input should return matching length"
    
    # Test single phase, multiple wavelengths
    phase = TEST_PHASES[0]
    waves = np.array(TEST_WAVES)
    m0_vec = salt3nir_m0(phase, waves)
    assert len(m0_vec) == len(waves), "Vectorized wavelength input should return matching length"
    
    # Test multiple phases and wavelengths
    m0_vec = salt3nir_m0(phases, waves)
    assert len(m0_vec) == len(phases), "Vectorized input should handle matching lengths" 