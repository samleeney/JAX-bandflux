"""Test bandflux calculations against SNCosmo."""
import numpy as np
import jax.numpy as jnp
import sncosmo
from jax_supernovae.core import Bandpass, HC_ERG_AA, MODEL_BANDFLUX_SPACING
from jax_supernovae.salt3nir import salt3nir_bandflux, salt3nir_m0, salt3nir_m1, salt3nir_colorlaw

def test_bandflux_matches_sncosmo():
    """Test that bandflux calculation matches SNCosmo exactly."""
    
    # Create SNCosmo model with SALT3-NIR
    snc_model = sncosmo.Model(source='salt3-nir')
    
    # Set up test parameters
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.1,
        'c': 0.2
    }
    snc_model.set(**params)
    
    # Test cases:
    # 1. Flat transmission bandpass
    print("\nTesting with flat transmission bandpass:")
    wave = np.linspace(4000, 8000, 101)  # Cover NIR range
    trans = np.ones_like(wave)  # Flat transmission
    band = Bandpass(wave, trans)
    snc_band = sncosmo.Bandpass(wave, trans)
    
    # Test at multiple phases
    phases = [-5.0, 0.0, 5.0, 10.0]
    for phase in phases:
        print(f"\nTesting phase {phase}")
        
        # Get fluxes directly from both implementations
        snc_flux = snc_model.bandflux(snc_band, phase)
        jax_flux = salt3nir_bandflux(phase, band, params)
        
        print("\nFinal bandflux:")
        print(f"  SNCosmo: {snc_flux}")
        print(f"  JAX:     {jax_flux}")
        print(f"  Ratio:   {jax_flux/snc_flux}")
        
        assert np.allclose(jax_flux, snc_flux, rtol=1e-3), \
            f"Fluxes don't match at phase {phase}"
    
    # 2. Test with real bandpasses
    print("\nTesting with Bessell bandpasses:")
    bands = ['bessellb', 'bessellv', 'bessellr', 'besselli']
    for band_name in bands:
        snc_band = sncosmo.get_bandpass(band_name)
        jax_band = Bandpass(snc_band.wave, snc_band.trans)
        
        for phase in phases:
            snc_flux = snc_model.bandflux(snc_band, phase)
            jax_flux = salt3nir_bandflux(phase, jax_band, params)
            
            print(f"{band_name:8s} Phase {phase:6.1f}: SNCosmo={snc_flux:10.3e}, "
                  f"JAX={jax_flux:10.3e}, Ratio={jax_flux/snc_flux:10.3f}")
            
            assert np.allclose(jax_flux, snc_flux, rtol=1e-3), \
                f"Fluxes don't match for {band_name} at phase {phase}"
    
    # 3. Test with different redshifts
    print("\nTesting with different redshifts:")
    band_name = 'bessellb'
    snc_band = sncosmo.get_bandpass(band_name)
    jax_band = Bandpass(snc_band.wave, snc_band.trans)
    
    redshifts = [0.0, 0.1, 0.5, 1.0]
    for z in redshifts:
        params['z'] = z
        snc_model.set(z=z)
        
        for phase in phases:
            snc_flux = snc_model.bandflux(snc_band, phase)
            jax_flux = salt3nir_bandflux(phase, jax_band, params)
            
            print(f"z={z:4.1f} Phase {phase:6.1f}: SNCosmo={snc_flux:10.3e}, "
                  f"JAX={jax_flux:10.3e}, Ratio={jax_flux/snc_flux:10.3f}")
            
            assert np.allclose(jax_flux, snc_flux, rtol=1e-3), \
                f"Fluxes don't match at z={z}, phase={phase}"
    
    # 4. Test with different model parameters
    print("\nTesting with different model parameters:")
    z = 0.1  # Fix redshift
    params['z'] = z
    snc_model.set(z=z)
    
    test_params = [
        {'x0': 1e-5, 'x1': 0.0, 'c': 0.0},  # Baseline
        {'x0': 2e-5, 'x1': 0.0, 'c': 0.0},  # Different x0
        {'x0': 1e-5, 'x1': 1.0, 'c': 0.0},  # Different x1
        {'x0': 1e-5, 'x1': 0.0, 'c': 0.3},  # Different c
    ]
    
    for p in test_params:
        # Update parameters
        test_p = params.copy()
        test_p.update(p)
        snc_model.set(**{k: v for k, v in p.items()})
        
        snc_flux = snc_model.bandflux(snc_band, 0.0)
        jax_flux = salt3nir_bandflux(0.0, jax_band, test_p)
        
        param_str = ", ".join(f"{k}={v}" for k, v in p.items())
        print(f"Params {param_str}: SNCosmo={snc_flux:10.3e}, "
              f"JAX={jax_flux:10.3e}, Ratio={jax_flux/snc_flux:10.3f}")
        
        assert np.allclose(jax_flux, snc_flux, rtol=1e-3), \
            f"Fluxes don't match with parameters: {param_str}"

if __name__ == '__main__':
    test_bandflux_matches_sncosmo() 