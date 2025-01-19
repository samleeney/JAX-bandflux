import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sncosmo
from jax_supernovae.models import Model
from jax_supernovae.salt2 import salt2_flux
from jax_supernovae.salt2_data import get_salt2_wave_grid

def test_bandflux_matches_sncosmo():
    """Test that our JAX implementation matches SNCosmo's output."""
    # Initialize original SNCosM model
    original_model = sncosmo.Model(source='salt2')
    
    # Initialize our JAX model
    jax_model = Model()
    jax_model.wave = get_salt2_wave_grid()
    jax_model._flux = lambda t, w: salt2_flux(t, w, jax_model.parameters)
    
    # Set the same parameters
    params = {'t0': 55000.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0, 'z': 0.5}
    original_model.set(**params)
    jax_model.parameters = params
    
    # Test at various times
    times = jnp.linspace(54950, 55050, 100)
    
    # Get fluxes from both implementations
    original_flux = original_model.bandflux('sdssg', times)
    jax_flux = jax_model.bandflux('sdssg', times)
    
    # Convert to numpy for comparison
    original_flux = np.array(original_flux)
    jax_flux = np.array(jax_flux)
    
    # Print some statistics about the differences
    print("\nComparison of SNCosmo vs JAX implementation:")
    print(f"Mean original flux: {np.mean(original_flux):.3e}")
    print(f"Mean JAX flux: {np.mean(jax_flux):.3e}")
    print(f"Max absolute difference: {np.max(np.abs(original_flux - jax_flux)):.3e}")
    print(f"Mean absolute difference: {np.mean(np.abs(original_flux - jax_flux)):.3e}")
    
    # Compute relative differences only where original flux is non-zero
    nonzero_mask = original_flux > 1e-10
    if np.any(nonzero_mask):
        rel_diff = np.abs((original_flux[nonzero_mask] - jax_flux[nonzero_mask]) / 
                         original_flux[nonzero_mask])
        print(f"Max relative difference (non-zero flux): {np.max(rel_diff):.3e}")
        print(f"Mean relative difference (non-zero flux): {np.mean(rel_diff):.3e}")
    
    # Compare results
    assert np.allclose(original_flux, jax_flux, rtol=1e-5), \
        "JAX implementation gives different results than SNCosmo" 