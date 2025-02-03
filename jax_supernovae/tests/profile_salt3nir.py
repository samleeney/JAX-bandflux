import cProfile
import pstats
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from jax_supernovae.salt3nir import (salt3nir_m0, salt3nir_m1, salt3nir_colorlaw,
                                    salt3nir_bandflux, salt3nir_multiband_flux)
from jax_supernovae.bandpasses import Bandpass
import time

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

def create_test_bandpass():
    """Create a test bandpass for profiling."""
    wave = jnp.linspace(3000, 9000, 1000)
    trans = jnp.ones_like(wave)
    return Bandpass(wave, trans)

def profile_m0():
    """Profile the M0 component function."""
    # Test with various input shapes
    phase = jnp.linspace(-10, 50, 100)
    wave = jnp.linspace(3000, 9000, 1000)
    
    # Compile function
    print("Compiling salt3nir_m0...")
    _ = salt3nir_m0(phase[0], wave[0])
    
    # Profile with multiple iterations
    print("Profiling salt3nir_m0...")
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        _ = salt3nir_m0(phase[:, None], wave[None, :])
    end = time.time()
    print(f"salt3nir_m0 took {(end - start)/n_iter:.4f} seconds per iteration")

def profile_m1():
    """Profile the M1 component function."""
    # Test with various input shapes
    phase = jnp.linspace(-10, 50, 100)
    wave = jnp.linspace(3000, 9000, 1000)
    
    # Compile function
    print("Compiling salt3nir_m1...")
    _ = salt3nir_m1(phase[0], wave[0])
    
    # Profile with multiple iterations
    print("Profiling salt3nir_m1...")
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        _ = salt3nir_m1(phase[:, None], wave[None, :])
    end = time.time()
    print(f"salt3nir_m1 took {(end - start)/n_iter:.4f} seconds per iteration")

def profile_colorlaw():
    """Profile the color law function."""
    wave = jnp.linspace(3000, 9000, 1000)
    
    # Compile function
    print("Compiling salt3nir_colorlaw...")
    _ = salt3nir_colorlaw(wave[0])
    
    # Profile with multiple iterations
    print("Profiling salt3nir_colorlaw...")
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        _ = salt3nir_colorlaw(wave)
    end = time.time()
    print(f"salt3nir_colorlaw took {(end - start)/n_iter:.4f} seconds per iteration")

def profile_bandflux():
    """Profile the bandflux function."""
    phase = jnp.linspace(-10, 50, 100)
    bandpass = create_test_bandpass()
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.1,
        'c': 0.1
    }
    
    # Compile function
    print("Compiling salt3nir_bandflux...")
    _ = salt3nir_bandflux(phase[0], bandpass, params)
    
    # Profile with multiple iterations
    print("Profiling salt3nir_bandflux...")
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        _ = salt3nir_bandflux(phase, bandpass, params)
    end = time.time()
    print(f"salt3nir_bandflux took {(end - start)/n_iter:.4f} seconds per iteration")

def profile_multiband_flux():
    """Profile the multiband flux function."""
    phase = jnp.linspace(-10, 50, 100)
    # Create bandpasses as a tuple instead of a list
    bandpasses = tuple(create_test_bandpass() for _ in range(5))
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.1,
        'c': 0.1
    }
    zps = jnp.ones(5) * 25.0
    
    # Compile function
    print("Compiling salt3nir_multiband_flux...")
    _ = salt3nir_multiband_flux(phase[0:1], bandpasses, params, zps=zps, zpsys='ab')
    
    # Profile with multiple iterations
    print("Profiling salt3nir_multiband_flux...")
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        _ = salt3nir_multiband_flux(phase, bandpasses, params, zps=zps, zpsys='ab')
    end = time.time()
    print(f"salt3nir_multiband_flux took {(end - start)/n_iter:.4f} seconds per iteration")

def main():
    """Run all profiling functions."""
    print("Starting profiling...")
    print("-" * 50)
    
    profile_m0()
    print("-" * 50)
    
    profile_m1()
    print("-" * 50)
    
    profile_colorlaw()
    print("-" * 50)
    
    profile_bandflux()
    print("-" * 50)
    
    profile_multiband_flux()

if __name__ == "__main__":
    main() 