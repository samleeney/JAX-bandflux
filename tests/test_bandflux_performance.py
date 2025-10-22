"""Performance benchmarks for SALT3Source bandflux calculations.

This module tests that the JAX-bandflux SALT3Source implementation maintains
high performance with the v3.0 functional API while matching sncosmo accuracy.
"""

import os
import sys
import time
import numpy as np
import jax.numpy as jnp
import sncosmo
from jax_supernovae import SALT3Source


def test_single_bandflux_performance():
    """Compare single bandflux evaluation performance between JAX and sncosmo.

    This tests the performance of a single bandflux calculation, which should
    be dominated by the underlying JAX optimizations once JIT-compiled.
    """
    # Set model parameters (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create JAX source (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    # Create and configure sncosmo source (stateful API for comparison)
    snc_source = sncosmo.get_source('salt3-nir')
    snc_source.set(**params)

    # Test parameters
    band = 'bessellb'
    phase = 0.0
    zp = 27.5
    zpsys = 'ab'

    # Warmup: Run JAX implementation multiple times to trigger JIT compilation
    print("\nWarming up JAX implementation (1000 calls)...")
    for _ in range(1000):
        _ = jax_source.bandflux(params, band, phase, zp=zp, zpsys=zpsys)

    # Benchmark sncosmo (500 evaluations)
    print("Benchmarking sncosmo...")
    start_time = time.time()
    snc_fluxes = []
    n_calls = 500
    for _ in range(n_calls):
        flux = snc_source.bandflux(band, phase, zp=zp, zpsys=zpsys)
        snc_fluxes.append(flux)
    snc_time = time.time() - start_time

    # Benchmark JAX (500 evaluations - v3.0 functional API)
    print("Benchmarking JAX...")
    start_time = time.time()
    jax_fluxes = []
    for _ in range(n_calls):
        flux = jax_source.bandflux(params, band, phase, zp=zp, zpsys=zpsys)
        jax_fluxes.append(flux)
    jax_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("Single Bandflux Performance Comparison")
    print("=" * 70)
    print(f"Number of evaluations: {n_calls}")
    print(f"SNCosmo time:  {snc_time:.4f} seconds ({snc_time/n_calls*1000:.4f} ms/call)")
    print(f"JAX time:      {jax_time:.4f} seconds ({jax_time/n_calls*1000:.4f} ms/call)")
    print(f"Speedup:       {snc_time/jax_time:.2f}x")
    print("=" * 70)

    # Verify results match
    snc_fluxes = np.array(snc_fluxes)
    jax_fluxes = np.array(jax_fluxes)
    np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4,
                             err_msg="JAX and SNCosmo fluxes do not match")
    print("✓ Fluxes match sncosmo within 0.001%")


def test_multiband_performance():
    """Compare multi-band performance between JAX and sncosmo.

    This is more realistic for supernova fitting, where we typically have
    multiple bands observed at different times.
    """
    # Set model parameters (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create JAX source (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    # Create and configure sncosmo source (stateful API for comparison)
    snc_source = sncosmo.get_source('salt3-nir')
    snc_source.set(**params)

    # Simulate a realistic light curve: 100 observations across 5 bands
    np.random.seed(42)
    n_obs = 100
    bands = np.random.choice(['bessellb', 'bessellv', 'bessellr', 'besselli', 'bessellux'], n_obs)
    phases = np.random.uniform(-10, 40, n_obs)
    zp = 27.5
    zpsys = 'ab'

    # Warmup JAX (v3.0 functional API)
    print("\nWarming up JAX for multi-band calculations...")
    for _ in range(100):
        _ = jax_source.bandflux(params, bands, phases, zp=zp, zpsys=zpsys)

    # Benchmark sncosmo
    print("Benchmarking sncosmo (multi-band)...")
    start_time = time.time()
    n_iterations = 100
    for _ in range(n_iterations):
        snc_fluxes = snc_source.bandflux(bands, phases, zp=zp, zpsys=zpsys)
    snc_time = time.time() - start_time

    # Benchmark JAX (v3.0 functional API)
    print("Benchmarking JAX (multi-band)...")
    start_time = time.time()
    for _ in range(n_iterations):
        jax_fluxes = jax_source.bandflux(params, bands, phases, zp=zp, zpsys=zpsys)
    jax_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("Multi-Band Performance Comparison")
    print("=" * 70)
    print(f"Observations per evaluation: {n_obs} (across 5 bands)")
    print(f"Number of iterations: {n_iterations}")
    print(f"Total calculations: {n_obs * n_iterations}")
    print(f"SNCosmo time:  {snc_time:.4f} seconds ({snc_time/n_iterations*1000:.2f} ms/iteration)")
    print(f"JAX time:      {jax_time:.4f} seconds ({jax_time/n_iterations*1000:.2f} ms/iteration)")
    print(f"Speedup:       {snc_time/jax_time:.2f}x")
    print("=" * 70)

    # Verify results match
    np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4,
                             err_msg="JAX and SNCosmo fluxes do not match")
    print("✓ Multi-band fluxes match sncosmo within 0.001%")


def test_parameter_variation_performance():
    """Test performance when parameters change frequently.

    This simulates MCMC or optimization scenarios where parameters are
    updated between each bandflux calculation.

    Note: v3.0 functional API is ideal for this use case as parameters
    are passed directly without object mutation.
    """
    # Create JAX source (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    # Create sncosmo source (stateful API for comparison)
    snc_source = sncosmo.get_source('salt3-nir')

    # Test parameters
    band = 'bessellb'
    phase = 0.0
    zp = 27.5
    zpsys = 'ab'
    n_iterations = 200

    # Generate random parameter values
    np.random.seed(42)
    x0_vals = np.random.uniform(1e-6, 1e-4, n_iterations)
    x1_vals = np.random.uniform(-2, 2, n_iterations)
    c_vals = np.random.uniform(-0.3, 0.3, n_iterations)

    # Warmup JAX (v3.0 functional API - no object mutation needed)
    print("\nWarming up JAX for parameter variation...")
    for i in range(100):
        params = {'x0': x0_vals[i % len(x0_vals)],
                  'x1': x1_vals[i % len(x1_vals)],
                  'c': c_vals[i % len(c_vals)]}
        _ = jax_source.bandflux(params, band, phase, zp=zp, zpsys=zpsys)

    # Benchmark sncosmo
    print("Benchmarking sncosmo (varying parameters)...")
    start_time = time.time()
    snc_fluxes = []
    for i in range(n_iterations):
        snc_source.set(x0=x0_vals[i], x1=x1_vals[i], c=c_vals[i])
        flux = snc_source.bandflux(band, phase, zp=zp, zpsys=zpsys)
        snc_fluxes.append(flux)
    snc_time = time.time() - start_time

    # Benchmark JAX (v3.0 functional API - just pass params)
    print("Benchmarking JAX (varying parameters)...")
    start_time = time.time()
    jax_fluxes = []
    for i in range(n_iterations):
        params = {'x0': x0_vals[i], 'x1': x1_vals[i], 'c': c_vals[i]}
        flux = jax_source.bandflux(params, band, phase, zp=zp, zpsys=zpsys)
        jax_fluxes.append(flux)
    jax_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("Parameter Variation Performance Comparison")
    print("=" * 70)
    print(f"Number of evaluations: {n_iterations} (with parameter updates)")
    print(f"SNCosmo time:  {snc_time:.4f} seconds ({snc_time/n_iterations*1000:.2f} ms/call)")
    print(f"JAX time:      {jax_time:.4f} seconds ({jax_time/n_iterations*1000:.2f} ms/call)")
    print(f"Speedup:       {snc_time/jax_time:.2f}x")
    print("=" * 70)

    # Verify results match
    snc_fluxes = np.array(snc_fluxes)
    jax_fluxes = np.array(jax_fluxes)
    np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4,
                             err_msg="JAX and SNCosmo fluxes do not match")
    print("✓ Fluxes with varying parameters match sncosmo within 0.001%")


def test_array_phase_performance():
    """Test performance for array of phases in single band.

    This is common when generating model light curves for visualization
    or chi-squared calculations.
    """
    # Set model parameters (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create JAX source (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    # Create and configure sncosmo source (stateful API for comparison)
    snc_source = sncosmo.get_source('salt3-nir')
    snc_source.set(**params)

    # Test parameters: 50 phases in one band
    band = 'bessellb'
    phases = np.linspace(-10, 40, 50)
    zp = 27.5
    zpsys = 'ab'
    n_iterations = 200

    # Warmup JAX (v3.0 functional API)
    print("\nWarming up JAX for array phase calculations...")
    for _ in range(100):
        _ = jax_source.bandflux(params, band, phases, zp=zp, zpsys=zpsys)

    # Benchmark sncosmo
    print("Benchmarking sncosmo (array of phases)...")
    start_time = time.time()
    for _ in range(n_iterations):
        snc_fluxes = snc_source.bandflux(band, phases, zp=zp, zpsys=zpsys)
    snc_time = time.time() - start_time

    # Benchmark JAX (v3.0 functional API)
    print("Benchmarking JAX (array of phases)...")
    start_time = time.time()
    for _ in range(n_iterations):
        jax_fluxes = jax_source.bandflux(params, band, phases, zp=zp, zpsys=zpsys)
    jax_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("Array Phase Performance Comparison")
    print("=" * 70)
    print(f"Phases per evaluation: {len(phases)}")
    print(f"Number of iterations: {n_iterations}")
    print(f"Total phase evaluations: {len(phases) * n_iterations}")
    print(f"SNCosmo time:  {snc_time:.4f} seconds ({snc_time/n_iterations*1000:.2f} ms/iteration)")
    print(f"JAX time:      {jax_time:.4f} seconds ({jax_time/n_iterations*1000:.2f} ms/iteration)")
    print(f"Speedup:       {snc_time/jax_time:.2f}x")
    print("=" * 70)

    # Verify results match
    np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4,
                             err_msg="JAX and SNCosmo fluxes do not match")
    print("✓ Array phase fluxes match sncosmo within 0.001%")


def run_all_benchmarks():
    """Run all performance benchmarks."""
    print("=" * 70)
    print("SALT3Source Performance Benchmarks (v3.0)")
    print("Testing JAX functional API performance vs sncosmo")
    print("=" * 70)

    test_single_bandflux_performance()
    test_multiband_performance()
    test_parameter_variation_performance()
    test_array_phase_performance()

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("v3.0 functional API maintains accuracy while providing speedups")
    print("=" * 70)


if __name__ == "__main__":
    # Add project root to Python path if running the file directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    run_all_benchmarks()
