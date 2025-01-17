import jax.numpy as jnp
import numpy as np
import sncosmo
from jax_supernovae.models import Model

def test_bandflux_matches_sncosmo():
    """Test that JAX implementation matches SNCosmo output."""
    # Create original SNCosmo model
    model_orig = sncosmo.Model("salt2")
    
    # Create JAX model
    model = Model("salt2")
    
    # Set parameters
    params = {
        "z": 0.5,
        "t0": 55000.0,
        "x0": 1e-5,
        "x1": 0.1,
        "c": 0.2
    }
    
    # Set parameters for both models
    model_orig.set(**params)  # Use keyword arguments for SNCosmo model
    model.parameters.update(params)  # Update parameters dictionary for JAX model
    
    # Test bandflux calculation
    time = jnp.array([55000.0, 55010.0])
    band = "sdssg"
    
    # Calculate bandflux using both implementations
    flux_orig = model_orig.bandflux(band, time)
    flux_jax = model.bandflux(band, time)
    
    # Compare results
    assert jnp.allclose(flux_orig, flux_jax, rtol=1e-5)
