"""Profile the SALT3 implementation with focus on JIT compilation effects."""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.profiler import start_trace, stop_trace, save_device_memory_profile
import tempfile
from functools import partial
import matplotlib.pyplot as plt

# Constants
H_ERG_S = 6.62607015e-27   # Planck constant in erg*s
C_AA_S = 2.99792458e18     # Speed of light in Angstrom/s
HC_ERG_AA = H_ERG_S * C_AA_S  # h*c in erg*Angstrom

# Add parent directory to path to import salt3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_supernovae.salt3 import (salt3_m0, salt3_m1, salt3_colorlaw,
                                salt3_bandflux, salt3_multiband_flux,
                                optimized_salt3_bandflux, optimized_salt3_multiband_flux,
                                precompute_bandflux_bridge)

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
    phase = jnp.linspace(-10, 50, n_phase)
    bandpasses = []
    bridges = []
    for i in range(n_bands):
        center_wave = 4000 + i * 1000
        bp = MockBandpass()
        bandpasses.append(bp)
        # Create bridge data for optimized functions
        bridge = precompute_bandflux_bridge(bp)
        bridges.append(bridge)
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.1,
        'c': 0.2
    }
    zps = jnp.ones(n_bands) * 25.0
    return phase, bandpasses, bridges, params, zps

# Create JIT-compiled versions of the functions
@jit
def jitted_m0(phase, wave):
    return salt3_m0(phase, wave)

@jit
def jitted_m1(phase, wave):
    return salt3_m1(phase, wave)

@jit
def jitted_colorlaw(wave):
    return salt3_colorlaw(wave)

# Create vectorised versions
vectorized_m0 = vmap(jitted_m0, in_axes=(0, None))
vectorized_m1 = vmap(jitted_m1, in_axes=(0, None))

def profile_compilation_overhead(phase, wave):
    """Profile the compilation overhead of JIT."""
    print("\nProfiling JIT compilation overhead...")
    
    # Time first call (includes compilation)
    start_time = time.time()
    _ = jitted_m0(phase[0], wave[0])
    compilation_time = time.time() - start_time
    print(f"First call (with compilation) M0: {compilation_time*1000:.2f} ms")
    
    # Time second call (no compilation)
    start_time = time.time()
    _ = jitted_m0(phase[0], wave[0])
    execution_time = time.time() - start_time
    print(f"Second call (no compilation) M0: {execution_time*1000:.2f} ms")
    print(f"Compilation overhead M0: {(compilation_time - execution_time)*1000:.2f} ms")

def profile_jitted_vs_unjitted(phase, wave, n_iterations=100):
    """Compare JIT-compiled vs non-JIT-compiled performance."""
    print("\nComparing JIT vs non-JIT performance...")
    
    # Profile non-JIT M0
    start_time = time.time()
    for _ in range(n_iterations):
        _ = salt3_m0(phase, wave)
    unjit_time = (time.time() - start_time) / n_iterations
    print(f"Non-JIT M0 computation time: {unjit_time*1000:.2f} ms")
    
    # Profile JIT M0
    start_time = time.time()
    for _ in range(n_iterations):
        _ = jitted_m0(phase, wave)
    jit_time = (time.time() - start_time) / n_iterations
    print(f"JIT M0 computation time: {jit_time*1000:.2f} ms")
    print(f"JIT speedup: {unjit_time/jit_time:.2f}x")
    
    # Profile vectorised M0
    start_time = time.time()
    for _ in range(n_iterations):
        _ = vectorized_m0(phase, wave)
    vec_time = (time.time() - start_time) / n_iterations
    print(f"Vectorised M0 computation time: {vec_time*1000:.2f} ms")
    print(f"Vectorisation speedup vs JIT: {jit_time/vec_time:.2f}x")

def profile_flux_calculations(phase, bandpasses, bridges, params, zps, n_iterations=100):
    """Profile flux calculations using optimized functions."""
    print("\nProfiling optimized flux calculations...")
    
    # Get first bandpass bridge data
    bridge = bridges[0]
    
    # Compile and warmup optimized_salt3_bandflux
    _ = optimized_salt3_bandflux(phase[0], bridge['wave'], bridge['dwave'], bridge['trans'], params, zp=zps[0], zpsys='ab')
    
    # Profile single bandpass
    start_time = time.time()
    for _ in range(n_iterations):
        _ = optimized_salt3_bandflux(phase, bridge['wave'], bridge['dwave'], bridge['trans'], params, zp=zps[0], zpsys='ab')
    single_time = (time.time() - start_time) / n_iterations
    print(f"Optimized single bandpass computation time: {single_time*1000:.2f} ms")
    
    # Profile multiband calculations
    print("\nProfiling optimized multiband calculations...")
    start_time = time.time()
    for _ in range(n_iterations):
        _ = optimized_salt3_multiband_flux(phase, bridges, params, zps=zps, zpsys='ab')
    multi_time = (time.time() - start_time) / n_iterations
    print(f"Optimized multiband computation time: {multi_time*1000:.2f} ms")

def plot_timing_results(timings, actual_time, compilation_times, save_path='timing_results.png'):
    """Create a bar plot of timing results.
    
    Args:
        timings (dict): Dictionary of component timings in seconds
        actual_time (float): Total execution time in seconds
        compilation_times (dict): Dictionary of compilation times in seconds
        save_path (str): Path to save the plot
    """
    # Convert times to milliseconds
    component_times = {k: v * 1000 for k, v in timings.items()}
    actual_time_ms = actual_time * 1000
    compilation_times_ms = {k: v * 1000 for k, v in compilation_times.items()}
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot component times
    components = list(component_times.keys())
    times = list(component_times.values())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(components)))
    
    # Left subplot: Component times (execution only)
    bars = ax1.bar(components, times, color=colors)
    ax1.set_title('Individual Component Times\n(Execution Only)')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom')
    
    # Middle subplot: Comparison of total component time vs actual time
    total_component_time = sum(times)
    comparison_times = [total_component_time, actual_time_ms]
    labels = ['Sum of\nComponents', 'Actual\nExecution']
    bars = ax2.bar(labels, comparison_times, color=['lightblue', 'lightgreen'])
    ax2.set_title('Total Time Comparison\n(Execution Only)')
    ax2.set_ylabel('Time (ms)')
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom')
    
    # Right subplot: Compilation times
    comp_names = list(compilation_times_ms.keys())
    comp_times = list(compilation_times_ms.values())
    bars = ax3.bar(comp_names, comp_times, color='salmon')
    ax3.set_title('Compilation Times')
    ax3.set_ylabel('Time (ms)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTiming plot saved to: {save_path}")

def profile_bandflux_components(phase, bridge, params, zp, n_iterations=1000):
    """Profile individual components of the bandflux calculation."""
    print("\nProfiling bandflux components...")
    
    # Extract parameters and bridge data
    wave = bridge['wave']
    dwave = bridge['dwave']
    trans = bridge['trans']
    z = params['z']
    t0 = params['t0']
    x0 = params['x0']
    x1 = params['x1']
    c = params['c']
    
    # Create timing dictionaries
    timings = {}
    compilation_times = {}
    
    # Time phase conversion (no compilation needed)
    start_time = time.time()
    for _ in range(n_iterations):
        a = 1.0 / (1.0 + z)
        restphase = (phase - t0) * a
        restwave = wave * a
    timings['phase_conversion'] = (time.time() - start_time) / n_iterations
    
    # Time color law with compilation
    start_time = time.time()
    _ = salt3_colorlaw(restwave)  # First call includes compilation
    compilation_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(n_iterations):
        cl = salt3_colorlaw(restwave)
    timings['colorlaw'] = (time.time() - start_time) / n_iterations
    compilation_times['colorlaw'] = compilation_time
    
    # Time M0 with compilation
    start_time = time.time()
    _ = salt3_m0(restphase[:, None], restwave[None, :])  # First call includes compilation
    compilation_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(n_iterations):
        m0 = salt3_m0(restphase[:, None], restwave[None, :])
    timings['m0'] = (time.time() - start_time) / n_iterations
    compilation_times['m0'] = compilation_time
    
    # Time M1 with compilation
    start_time = time.time()
    _ = salt3_m1(restphase[:, None], restwave[None, :])  # First call includes compilation
    compilation_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(n_iterations):
        m1 = salt3_m1(restphase[:, None], restwave[None, :])
    timings['m1'] = (time.time() - start_time) / n_iterations
    compilation_times['m1'] = compilation_time
    
    # Time flux calculation and integration
    start_time = time.time()
    for _ in range(n_iterations):
        rest_flux = x0 * (m0 + x1 * m1) * 10**(-0.4 * cl[None, :] * c) * a
        integrand = wave[None, :] * trans[None, :] * rest_flux
        result = jnp.trapezoid(integrand, wave, axis=1)
    timings['flux_and_integration'] = (time.time() - start_time) / n_iterations
    
    # Time zero point calculation
    start_time = time.time()
    for _ in range(n_iterations):
        zpbandflux = 3631e-23 * dwave / 6.62607015e-27 * jnp.sum(trans / wave)
        zpnorm = 10**(0.4 * zp) / zpbandflux
        result = result * zpnorm
    timings['zeropoint'] = (time.time() - start_time) / n_iterations
    
    # Calculate total time
    total_time = sum(timings.values())
    
    # Print results
    print("\nComponent-wise timing breakdown (execution only):")
    print("-" * 50)
    for component, timing in timings.items():
        ms_time = timing * 1000
        percentage = (timing / total_time) * 100
        print(f"{component:20s}: {ms_time:6.2f} ms ({percentage:5.1f}%)")
    print("-" * 50)
    print(f"Total component time: {total_time*1000:.2f} ms")
    
    # Create a fresh JIT-compiled version of the complete function to measure true compilation time
    @jax.jit
    def complete_bandflux(phase, wave, dwave, trans, params, zp):
        # Convert inputs to arrays and check if input was scalar
        phase = jnp.atleast_1d(phase)
        is_scalar = len(phase.shape) == 0

        z = params['z']
        t0 = params['t0']
        x0 = params['x0']
        x1 = params['x1']
        c = params['c']

        # Calculate scaling factor and transform phase to rest-frame.
        a = 1.0 / (1.0 + z)
        restphase = (phase - t0) * a
        restwave = wave * a
        
        # Compute colour law on the restwave grid.
        cl = salt3_colorlaw(restwave)

        # Compute m0 and m1 components over the 2D grid.
        m0 = salt3_m0(restphase[:, None], restwave[None, :])
        m1 = salt3_m1(restphase[:, None], restwave[None, :])

        # Compute rest-frame flux including the colour law effect.
        rest_flux = x0 * (m0 + x1 * m1) * 10**(-0.4 * cl[None, :] * c) * a
        integrand = wave[None, :] * trans[None, :] * rest_flux
        result = jnp.trapezoid(integrand, wave, axis=1) / HC_ERG_AA

        # Apply zero point correction
        zpbandflux = 3631e-23 * dwave / H_ERG_S * jnp.sum(trans / wave)
        zpnorm = 10**(0.4 * zp) / zpbandflux
        result = result * zpnorm

        # Return scalar if input was scalar
        if is_scalar:
            result = result[0]
        return result
    
    # Time complete function compilation
    start_time = time.time()
    _ = complete_bandflux(phase, wave, dwave, trans, params, zp)
    compilation_times['complete_function'] = time.time() - start_time
    
    # Time complete function execution
    start_time = time.time()
    for _ in range(n_iterations):
        _ = complete_bandflux(phase, wave, dwave, trans, params, zp)
    actual_time = (time.time() - start_time) / n_iterations
    print(f"Actual complete function time (execution only): {actual_time*1000:.2f} ms")
    
    print("\nCompilation times:")
    print("-" * 50)
    for component, timing in compilation_times.items():
        print(f"{component:20s}: {timing*1000:6.2f} ms")
    
    # Create plot of timing results
    plot_timing_results(timings, actual_time, compilation_times)
    
    return timings, actual_time, compilation_times

def main():
    # Create test data
    print("Creating test data...")
    phase, bandpasses, bridges, params, zps = create_test_data()
    wave = bandpasses[0].integration_wave
    
    # Create temporary directory for profiling output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Start JAX profiling
        print("\nStarting JAX profiling...")
        start_trace(tmpdir)
        
        # Profile compilation overhead
        profile_compilation_overhead(phase, wave)
        
        # Profile JIT vs non-JIT
        profile_jitted_vs_unjitted(phase, wave)
        
        # Profile optimized flux calculations
        profile_flux_calculations(phase, bandpasses, bridges, params, zps)
        
        # Profile bandflux components
        profile_bandflux_components(phase, bridges[0], params, zps[0])
        
        # Stop profiling and save memory profile
        stop_trace()
        save_device_memory_profile(os.path.join(tmpdir, "memory_profile.prof"))
        
        print(f"\nProfiling data saved to: {tmpdir}")
        print("You can view the profiling data using TensorBoard:")
        print(f"tensorboard --logdir={tmpdir}")

if __name__ == "__main__":
    main() 