import os
import sys
import time
import numpy as np
import jax.numpy as jnp
import sncosmo
from jax_supernovae.salt3 import salt3_bandflux
from jax_supernovae.bandpasses import Bandpass

def test_bandflux_performance():
    """Compare performance between sncosmo and jax bandflux implementations."""
    # Set model parameters
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create sncosmo model and set parameters
    snc_model = sncosmo.Model(source='salt3-nir')
    snc_model.update(params)

    # Create a test bandpass
    wave = np.linspace(3000.0, 9000.0, 100)
    trans = np.ones_like(wave)
    snc_band = sncosmo.Bandpass(wave, trans, name='test_band')
    jax_band = Bandpass(wave, trans)

    # Use a single phase for testing
    phase = jnp.array(0.0)

    # First run JAX implementation 5000 times for warmup
    print("\nWarming up JAX (5000 calls)...")
    for _ in range(5000):
        _ = salt3_bandflux(phase, jax_band, params)
    
    print("Starting performance comparison...")
    
    # Time sncosmo implementation (5000 calls)
    start_time = time.time()
    snc_fluxes = []
    for _ in range(500):
        flux = snc_model.bandflux(snc_band, float(phase))
        snc_fluxes.append(flux)
    snc_time = time.time() - start_time
    snc_fluxes = np.array(snc_fluxes)

    # Time JAX implementation with additional warmup
    # First do 1000 warmup calls that we'll ignore
    for _ in range(1000):
        _ = salt3_bandflux(phase, jax_band, params)
    
    # Now measure the actual performance
    start_time = time.time()
    jax_fluxes = []
    for _ in range(500):
        flux = salt3_bandflux(phase, jax_band, params)
        jax_fluxes.append(float(flux[0]))
    jax_time = time.time() - start_time
    jax_fluxes = np.array(jax_fluxes)

    # Print results
    print("\nPerformance Comparison (5000 bandflux evaluations, after warmup):")
    print("-" * 60)
    print(f"SNCosmo time: {snc_time:.4f} seconds")
    print(f"JAX time:     {jax_time:.4f} seconds")
    print(f"Speedup:      {snc_time/jax_time:.2f}x")
    print(f"Average time per call:")
    print(f"  SNCosmo: {(snc_time/5000)*1000:.4f} ms")
    print(f"  JAX:     {(jax_time/5000)*1000:.4f} ms")
    print("-" * 60)

    # Verify results match
    np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-2,
                             err_msg="JAX and SNCosmo fluxes do not match")

if __name__ == "__main__":
    # Add project root to Python path if running the file directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    test_bandflux_performance() 