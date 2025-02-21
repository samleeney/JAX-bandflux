"""Profile the SALT3 implementation."""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.profiler import start_trace, stop_trace, save_device_memory_profile
import tempfile
import matplotlib.pyplot as plt

# Add parent directory to path to import salt3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_supernovae.salt3 import (
    salt3_m0, salt3_m1, salt3_colorlaw,
    salt3_bandflux, salt3_multiband_flux,
    optimized_salt3_bandflux, optimized_salt3_multiband_flux,
    precompute_bandflux_bridge,
    H_ERG_S, HC_ERG_AA
)

# Output directory
PROFILE_DIR = os.path.dirname(os.path.abspath(__file__))

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

def profile_flux_calculations(phase, bandpasses, bridges, params, zps, n_iterations=100):
    """Profile flux calculations using optimized functions."""
    print("\nProfiling optimized flux calculations...")
    
    # Get first bandpass bridge data
    bridge = bridges[0]
    
    # Warmup call
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

def plot_function_timings(phase, bandpasses, bridges, params, zps, n_iterations=100, n_runs=100):
    """Profile and plot timings of all functions.
    
    Args:
        phase (array): Phase values to test with
        bandpasses (list): List of bandpass objects
        bridges (list): List of bridge data for each bandpass
        params (dict): Model parameters
        zps (array): Zero points for each bandpass
        n_iterations (int): Number of iterations for timing
        n_runs (int): Number of times to repeat timing measurements
    """
    print("\nProfiling all functions...")
    wave = bandpasses[0].integration_wave
    bridge = bridges[0]
    
    # Dictionary to store timings and statistics
    timings = {}
    timing_stats = {}  # Will store mean and std for each function
    compilation_times = {}
    
    def time_function(func, args, name):
        """Helper function to time a function multiple times."""
        # First compilation and warmup
        if name not in compilation_times:
            start_time = time.time()
            _ = func(*args)
            compilation_times[name] = time.time() - start_time
            # Warmup call
            _ = func(*args)
        
        # Multiple timing runs
        run_times = np.zeros(n_runs)
        for run in range(n_runs):
            start_time = time.time()
            for _ in range(n_iterations):
                _ = func(*args)
            run_times[run] = (time.time() - start_time) / n_iterations
        
        # Calculate statistics
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        timings[name] = run_times
        timing_stats[name] = {'mean': mean_time, 'std': std_time}
    
    # Time each function
    time_function(salt3_m0, (phase[0], wave[0]), 'm0')
    time_function(salt3_m1, (phase[0], wave[0]), 'm1')
    time_function(salt3_colorlaw, (wave[0],), 'colorlaw')
    time_function(
        optimized_salt3_bandflux, 
        (phase[0], bridge['wave'], bridge['dwave'], bridge['trans'], params, zps[0], 'ab'),
        'single_bandflux'
    )
    time_function(
        optimized_salt3_multiband_flux,
        (phase, bridges, params, zps, 'ab'),
        'multiband_flux'
    )
    
    # Convert times to milliseconds for plotting
    timing_stats_ms = {
        k: {
            'mean': v['mean'] * 1000,
            'std': v['std'] * 1000
        } for k, v in timing_stats.items()
    }
    compilation_times_ms = {k: v * 1000 for k, v in compilation_times.items()}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution times with error bars
    functions = list(timing_stats_ms.keys())
    means = [timing_stats_ms[f]['mean'] for f in functions]
    stds = [timing_stats_ms[f]['std'] for f in functions]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(functions)))
    
    bars = ax1.bar(functions, means, yerr=stds, capsize=5, color=colors)
    ax1.set_title('Function Execution Times\n(per call, averaged over {} runs)'.format(n_runs))
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on the bars
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}±{std:.3f}ms',
                ha='center', va='bottom')
    
    # Plot compilation times
    functions = list(compilation_times_ms.keys())
    times = list(compilation_times_ms.values())
    
    bars = ax2.bar(functions, times, color='salmon')
    ax2.set_title('Compilation Times\n(first call)')
    ax2.set_ylabel('Time (ms)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom')
    
    # Print timing statistics
    print("\nExecution times (per call):")
    print("-" * 70)
    print(f"{'Function':20s} {'Mean (ms)':10s} {'Std (ms)':10s} {'CV (%)':10s}")
    print("-" * 70)
    for func in functions:
        mean = timing_stats_ms[func]['mean']
        std = timing_stats_ms[func]['std']
        cv = (std / mean) * 100 if mean > 0 else 0
        print(f"{func:20s} {mean:10.3f} {std:10.3f} {cv:10.1f}")
    
    print("\nCompilation times:")
    print("-" * 50)
    for func, time_ms in compilation_times_ms.items():
        print(f"{func:20s}: {time_ms:.2f} ms")
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = os.path.join(PROFILE_DIR, 'salt3_function_timings.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTiming plot saved to: {output_file}")
    
    return timing_stats, compilation_times

def main():
    # Create test data
    print("Creating test data...")
    phase, bandpasses, bridges, params, zps = create_test_data()
    
    # Create temporary directory for profiling output
    profile_tmp_dir = os.path.join(PROFILE_DIR, 'tmp')
    os.makedirs(profile_tmp_dir, exist_ok=True)
    
    # Start JAX profiling
    print("\nStarting JAX profiling...")
    start_trace(profile_tmp_dir)
    
    # Profile optimized flux calculations
    profile_flux_calculations(phase, bandpasses, bridges, params, zps)
    
    # Profile and plot all function timings
    plot_function_timings(phase, bandpasses, bridges, params, zps)
    
    # Stop profiling and save memory profile
    stop_trace()
    memory_profile_path = os.path.join(PROFILE_DIR, "salt3_memory_profile.prof")
    save_device_memory_profile(memory_profile_path)
    
    print(f"\nProfiling data saved to: {PROFILE_DIR}")
    print("You can view the profiling data using TensorBoard:")
    print(f"tensorboard --logdir={profile_tmp_dir}")

if __name__ == "__main__":
    main() 