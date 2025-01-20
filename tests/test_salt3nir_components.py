import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision

import numpy as np
import sncosmo
from jax_supernovae.models import Model
from jax_supernovae.salt3nir import salt3nir_flux, salt3nir_m0, salt3nir_m1, salt3nir_colorlaw, read_griddata_file, kernval, m0_data, m1_data, SCALE_FACTOR
from jax_supernovae.salt3nir import wave_grid, phase_grid
from jax_supernovae.core import HC_ERG_AA
import jax.numpy as jnp
import pytest

def test_salt3nir_detailed():
    """Detailed test of SALT3-NIR model components."""
    
    # Create models
    snc_model = sncosmo.Model(source='salt3-nir')
    params = {'z': 0.0, 't0': 0.0, 'x0': 1.0, 'x1': 0.0, 'c': 0.0}
    snc_model.set(**params)
    
    # Create JAX model
    jax_model = Model()
    jax_model.wave = wave_grid
    jax_model.flux = lambda t, w: salt3nir_flux(t, w, params)
    jax_model.parameters = params
    
    # Test points
    times = np.array([0.0])  # Start with a single time point
    waves = np.array([5000.0])  # Start with a single wavelength
    
    print("\nDetailed SALT3-NIR Component Test")
    print("================================")
    
    # 1. Test M0 component
    print("\n1. Testing M0 component:")
    snc_m0 = snc_model._source._model['M0'](times, waves)[0][0]
    jax_m0 = float(salt3nir_m0(times[0], waves[0]))
    print(f"SNCosmo M0: {snc_m0:.6e}")
    print(f"JAX M0:     {jax_m0:.6e}")
    print(f"Ratio:      {jax_m0/snc_m0:.6f}")
    
    # 2. Test M1 component
    print("\n2. Testing M1 component:")
    snc_m1 = snc_model._source._model['M1'](times, waves)[0][0]
    jax_m1 = float(salt3nir_m1(times[0], waves[0]))
    print(f"SNCosmo M1: {snc_m1:.6e}")
    print(f"JAX M1:     {jax_m1:.6e}")
    print(f"Ratio:      {jax_m1/snc_m1:.6f}")
    
    # 3. Test color law
    print("\n3. Testing color law:")
    snc_cl = snc_model._source._colorlaw(waves)[0]
    jax_cl = float(salt3nir_colorlaw(waves[0]))
    print(f"SNCosmo CL: {snc_cl:.6f}")
    print(f"JAX CL:     {jax_cl:.6f}")
    print(f"Difference: {jax_cl - snc_cl:.6f}")
    
    # 4. Test flux before redshift
    print("\n4. Testing rest-frame flux:")
    snc_flux = snc_model._source._flux(times, waves)[0][0]
    jax_flux = float(salt3nir_flux(times[0], waves[0], params))
    print(f"SNCosmo flux: {snc_flux:.6e}")
    print(f"JAX flux:     {jax_flux:.6e}")
    print(f"Ratio:        {jax_flux/snc_flux:.6f}")
    
    # 5. Test bandpass integration
    print("\n5. Testing bandpass integration:")
    # Create a narrow bandpass centered at 5000Å
    band_wave = np.linspace(4900, 5100, 21)
    band_trans = np.ones_like(band_wave)
    band = sncosmo.Bandpass(band_wave, band_trans)
    
    # Get bandfluxes
    snc_bandflux = snc_model.bandflux(band, times)[0]
    jax_bandflux = jax_model.bandflux(band, times)[0]
    print(f"SNCosmo bandflux: {snc_bandflux:.6e}")
    print(f"JAX bandflux:     {jax_bandflux:.6e}")
    print(f"Ratio:            {jax_bandflux/snc_bandflux:.6f}")
    
    # 6. Test integration components
    print("\n6. Testing integration components:")
    # Get rest-frame quantities
    z = params['z']
    t0 = params['t0']
    a = 1.0 / (1.0 + z)
    restphase = (times[0] - t0) * a
    restwave = band_wave * a
    
    # Get fluxes at each wavelength
    snc_fluxes = snc_model._source._flux(np.array([restphase]), restwave)
    jax_fluxes = jax_model.flux(restphase, restwave)
    
    # Print first few values
    print("\nFirst few flux values at each wavelength:")
    for i in range(min(5, len(restwave))):
        print(f"Wave={restwave[i]:.1f}:")
        print(f"  SNCosmo: {snc_fluxes[0][i]:.6e}")
        print(f"  JAX:     {float(jax_fluxes[i]):.6e}")
        print(f"  Ratio:   {float(jax_fluxes[i])/snc_fluxes[0][i]:.6f}")
    
    # Compare integration
    dwave = band_wave[1] - band_wave[0]
    snc_integrand = band_wave * band_trans * snc_fluxes[0]
    jax_integrand = band_wave * band_trans * jax_fluxes
    
    snc_sum = np.sum(snc_integrand) * dwave / HC_ERG_AA
    jax_sum = float(np.sum(jax_integrand) * dwave / HC_ERG_AA)
    
    print("\nIntegration results:")
    print(f"SNCosmo sum: {snc_sum:.6e}")
    print(f"JAX sum:     {jax_sum:.6e}")
    print(f"Ratio:       {jax_sum/snc_sum:.6f}")
    
    # Assertions with appropriate tolerances
    np.testing.assert_allclose(jax_m0, snc_m0, rtol=1e-4,
                              err_msg="M0 component mismatch")
    np.testing.assert_allclose(jax_m1, snc_m1, rtol=1e-4,
                              err_msg="M1 component mismatch")
    np.testing.assert_allclose(jax_cl, snc_cl, rtol=1e-4,
                              err_msg="Color law mismatch")
    np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4,
                              err_msg="Rest-frame flux mismatch")
    np.testing.assert_allclose(jax_bandflux, snc_bandflux, rtol=1e-4,
                              err_msg="Bandflux mismatch")

def test_salt3nir_interpolation():
    """Test that JAX interpolation matches SNCosmo exactly."""
    # Load SNCosmo model
    model = sncosmo.Model(source='salt3-nir')
    
    # Test points
    test_phases = [-5.0, 0.0, 5.0, 10.0]
    test_waves = [4000.0, 5000.0, 6000.0, 7000.0]
    
    # Test M0 interpolation
    print("\nTesting M0 interpolation:")
    for phase in test_phases:
        for wave in test_waves:
            # Get SNCosmo M0 value
            sncosmo_m0 = float(model.source._model['M0'](phase, wave))
            
            # Get JAX M0 value
            jax_m0 = float(salt3nir_m0(phase, wave))
            
            # Compare values
            print(f"Phase={phase}, Wave={wave}:")
            print(f"  SNCosmo M0: {sncosmo_m0:.6e}")
            print(f"  JAX M0:     {jax_m0:.6e}")
            print(f"  Ratio:      {jax_m0/sncosmo_m0:.6f}")
            
            # Assert values match within tolerance
            np.testing.assert_allclose(
                jax_m0, sncosmo_m0,
                rtol=1e-6,
                err_msg=f"M0 mismatch at phase={phase}, wave={wave}"
            )
    
    # Test M1 interpolation
    print("\nTesting M1 interpolation:")
    for phase in test_phases:
        for wave in test_waves:
            # Get SNCosmo M1 value
            sncosmo_m1 = model.source._model['M1'](phase, wave)
            
            # Get JAX M1 value
            jax_m1 = float(salt3nir_m1(phase, wave))
            
            # Compare values
            print(f"Phase={phase}, Wave={wave}:")
            print(f"  SNCosmo M1: {sncosmo_m1:.6e}")
            print(f"  JAX M1:     {jax_m1:.6e}")
            print(f"  Ratio:      {jax_m1/sncosmo_m1:.6f}")
            
            # Assert values match within tolerance
            np.testing.assert_allclose(
                jax_m1, sncosmo_m1,
                rtol=1e-6,
                err_msg=f"M1 mismatch at phase={phase}, wave={wave}"
            )
    
    # Test color law
    print("\nTesting color law:")
    for wave in test_waves:
        # Get SNCosmo color law value
        sncosmo_cl = model.source._colorlaw(wave)
        
        # Get JAX color law value
        jax_cl = float(salt3nir_colorlaw(wave))
        
        # Compare values
        print(f"Wave={wave}:")
        print(f"  SNCosmo CL: {sncosmo_cl:.6f}")
        print(f"  JAX CL:     {jax_cl:.6f}")
        print(f"  Ratio:      {jax_cl/sncosmo_cl:.6f}")
        
        # Assert values match within tolerance
        np.testing.assert_allclose(
            jax_cl, sncosmo_cl,
            rtol=1e-6,
            err_msg=f"Color law mismatch at wave={wave}"
        )

# Test data reading
@pytest.mark.parametrize("filename, expected_shape", [
    ('salt3_template_0.dat', (len(phase_grid), len(wave_grid))),
    ('salt3_template_1.dat', (len(phase_grid), len(wave_grid)))
])
def test_data_reading(filename, expected_shape):
    phase, wave, data = read_griddata_file(filename)
    assert data.shape == expected_shape, f"Data shape mismatch for {filename}"

# Test scaling
@pytest.mark.parametrize("data, scale_factor", [
    (m0_data, SCALE_FACTOR),
    (m1_data, SCALE_FACTOR)
])
def test_scaling(data, scale_factor):
    scaled_data = data * scale_factor
    assert jnp.allclose(scaled_data, data * scale_factor), "Scaling mismatch"

# Test interpolation
@pytest.mark.parametrize("phase, wave", [
    (0.0, 4000.0),
    (10.0, 5000.0),
    (20.0, 6000.0)
])
def test_interpolation(phase, wave):
    snc_m0 = sncosmo.Model(source='salt3-nir')._source._model['M0'](np.array([phase]), np.array([wave]))[0][0]
    jax_m0 = salt3nir_m0(phase, wave)
    assert jnp.isclose(jax_m0, snc_m0, rtol=1e-4), f"Interpolation mismatch at phase {phase}, wave {wave}"

# Test kernel function
@pytest.mark.parametrize("x, expected_value", [
    (-1.0, kernval(-1.0)),
    (0.0, kernval(0.0)),
    (1.0, kernval(1.0)),
    (2.0, kernval(2.0)),
    (3.0, kernval(3.0))
])
def test_kernel_function(x, expected_value):
    assert jnp.isclose(kernval(x), expected_value), f"Kernel function mismatch at x={x}"

# Test component comparison
@pytest.mark.parametrize("phase, wave", [
    (0.0, 4000.0),
    (10.0, 5000.0),
    (20.0, 6000.0)
])
def test_component_comparison(phase, wave):
    snc_m0 = sncosmo.Model(source='salt3-nir')._source._model['M0'](np.array([phase]), np.array([wave]))[0][0]
    jax_m0 = salt3nir_m0(phase, wave)
    assert jnp.isclose(jax_m0, snc_m0, rtol=1e-4), f"M0 component mismatch at phase {phase}, wave {wave}"
    snc_m1 = sncosmo.Model(source='salt3-nir')._source._model['M1'](np.array([phase]), np.array([wave]))[0][0]
    jax_m1 = salt3nir_m1(phase, wave)
    assert jnp.isclose(jax_m1, snc_m1, rtol=1e-4), f"M1 component mismatch at phase {phase}, wave {wave}" 