"""Profile the SALT3 implementation using JAX's profiling tools."""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.profiler import start_trace, stop_trace, save_device_memory_profile
import tempfile

# Add parent directory to path to import salt3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_supernovae.salt3 import (salt3_m0, salt3_m1, salt3_colorlaw,
                                salt3_bandflux, salt3_multiband_flux,
                                optimized_salt3_bandflux, optimized_salt3_multiband_flux)

# Mock bandpass class for testing
class MockBandpass:
    def __init__(self, wave_min=3000, wave_max=7000, n_points=200):
        self.integration_wave = jnp.linspace(wave_min, wave_max, n_points)
        self.integration_spacing = (wave_max - wave_min) / (n_points - 1)
        
    def __call__(self, wave):
        # Simple gaussian transmission function
        return jnp.exp(-(wave - 5000)**2 / (2 * 500**2))

def create_test_data(n_phase=100, n_bands=5):
    """Create test data for profiling."""
    # Create phase array
    phase = jnp.linspace(-10, 50, n_phase)
    
    # Create bandpasses with different central wavelengths
    bandpasses = []
    for i in range(n_bands):
        center_wave = 4000 + i * 1000  # Space them out by 1000Å
        bp = MockBandpass()
        bandpasses.append(bp)
    
    # Create parameters
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.1,
        'c': 0.2
    }
    
    # Create zero points
    zps = jnp.ones(n_bands) * 25.0
    
    return phase, bandpasses, params, zps

def profile_individual_components(phase, wave, n_iterations=100):
    """Profile individual model components."""
    print("\nProfiling individual components...")
    
    # Compile functions first
    _ = salt3_m0(phase[0], wave[0])
    _ = salt3_m1(phase[0], wave[0])
    _ = salt3_colorlaw(wave[0])
    
    # Profile M0 component
    start_time = time.time()
    for _ in range(n_iterations):
        _ = salt3_m0(phase, wave)
    m0_time = (time.time() - start_time) / n_iterations
    print(f"Average M0 computation time: {m0_time*1000:.2f} ms")
    
    # Profile M1 component
    start_time = time.time()
    for _ in range(n_iterations):
        _ = salt3_m1(phase, wave)
    m1_time = (time.time() - start_time) / n_iterations
    print(f"Average M1 computation time: {m1_time*1000:.2f} ms")
    
    # Profile color law
    start_time = time.time()
    for _ in range(n_iterations):
        _ = salt3_colorlaw(wave)
    cl_time = (time.time() - start_time) / n_iterations
    print(f"Average color law computation time: {cl_time*1000:.2f} ms")

def profile_flux_calculations(phase, bandpasses, params, zps, n_iterations=100):
    """Profile flux calculation functions."""
    print("\nProfiling flux calculations...")
    
    # Compile function first
    _ = salt3_bandflux(phase[0], bandpasses[0], params, zp=zps[0], zpsys='ab')
    
    # Profile single bandpass flux
    start_time = time.time()
    for _ in range(n_iterations):
        _ = salt3_bandflux(phase, bandpasses[0], params, zp=zps[0], zpsys='ab')
    single_time = (time.time() - start_time) / n_iterations
    print(f"Average single bandpass flux computation time: {single_time*1000:.2f} ms")
    
    # Profile individual multiband calculations
    print("\nProfiling individual multiband calculations...")
    start_time = time.time()
    for _ in range(n_iterations):
        results = []
        for i, bp in enumerate(bandpasses):
            result = salt3_bandflux(phase, bp, params, zp=zps[i], zpsys='ab')
            results.append(result)
        _ = jnp.stack(results, axis=1)
    multi_time = (time.time() - start_time) / n_iterations
    print(f"Average multiband flux computation time: {multi_time*1000:.2f} ms")

def main():
    # Create test data
    print("Creating test data...")
    phase, bandpasses, params, zps = create_test_data()
    wave = bandpasses[0].integration_wave
    
    # Create temporary directory for profiling output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Start JAX profiling
        print("\nStarting JAX profiling...")
        start_trace(tmpdir)
        
        # Profile individual components
        profile_individual_components(phase, wave)
        
        # Profile flux calculations
        profile_flux_calculations(phase, bandpasses, params, zps)
        
        # Stop profiling and save memory profile
        stop_trace()
        save_device_memory_profile(os.path.join(tmpdir, "memory_profile.prof"))
        
        print(f"\nProfiling data saved to: {tmpdir}")
        print("You can view the profiling data using TensorBoard:")
        print(f"tensorboard --logdir={tmpdir}")

if __name__ == "__main__":
    main() 