import jax
import jax.numpy as jnp
import numpy as np
import time
import cProfile
import pstats
import matplotlib.pyplot as plt
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.data import load_and_process_data

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

@jax.jit
def compute_likelihood(params, times, fluxes, fluxerrs, bridges, zps, band_indices):
    """Compute likelihood for a single set of parameters."""
    t0, log_x0, x1, c = params
    z = 0.1  # Fixed redshift for testing
    x0 = 10 ** log_x0
    
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    
    # Calculate model fluxes
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    
    # Compute chi-squared
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs) ** 2)
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * fluxerrs ** 2)))
    
    return log_likelihood

def plot_timing_histogram(times, output_file='profile/timing_histogram.png'):
    """Create and save a histogram of iteration times."""
    plt.figure(figsize=(12, 7))
    
    # Calculate optimal number of bins using Freedman-Diaconis rule
    q75, q25 = np.percentile(times, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(times) ** (1/3))
    n_bins = int(np.ceil((times.max() - times.min()) / bin_width)) if bin_width > 0 else 50
    
    # Plot histogram
    plt.hist(times * 1000, bins=min(n_bins, 100), edgecolor='black', alpha=0.7)  # Convert to milliseconds
    plt.xlabel('Iteration Time (milliseconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Likelihood Computation Times (10,000 iterations)')
    
    # Add mean and std dev lines
    mean_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    median_time = np.median(times) * 1000
    
    plt.axvline(mean_time, color='r', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_time:.3f} ms')
    plt.axvline(median_time, color='b', linestyle='dashed', linewidth=2, 
                label=f'Median: {median_time:.3f} ms')
    plt.axvline(mean_time + std_time, color='g', linestyle=':', linewidth=2, 
                label=f'Mean ± Std: {std_time:.3f} ms')
    plt.axvline(mean_time - std_time, color='g', linestyle=':', linewidth=2)
    
    # Add percentile information
    percentiles = np.percentile(times * 1000, [5, 25, 50, 75, 95])
    plt.text(0.98, 0.95, 
            f'Percentiles (ms):\n' + 
            f'5th:  {percentiles[0]:.3f}\n' +
            f'25th: {percentiles[1]:.3f}\n' +
            f'50th: {percentiles[2]:.3f}\n' +
            f'75th: {percentiles[3]:.3f}\n' +
            f'95th: {percentiles[4]:.3f}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"\nHistogram saved as '{output_file}'")

def profile_likelihood(n_iterations=10000, warmup=5):
    """Profile likelihood computation."""
    print(f"Profiling likelihood computation for {n_iterations} iterations (with {warmup} warmup iterations)")
    
    # Load data
    print("\nLoading data...")
    times, fluxes, fluxerrs, zps, band_indices, bridges, _ = load_and_process_data('19dwz', data_dir='data', fix_z=True)
    
    # Example parameters
    test_params = jnp.array([58500.0, -3.0, 0.0, 0.0])  # t0, log_x0, x1, c
    
    # JIT compile and warmup
    print("\nWarming up JIT compilation...")
    likelihood_fn = lambda p: compute_likelihood(p, times, fluxes, fluxerrs, bridges, zps, band_indices)
    
    for i in range(warmup):
        print(f"Warmup iteration {i+1}/{warmup}")
        _ = likelihood_fn(test_params)
    
    # Profiling
    print("\nStarting profiling...")
    iteration_times = []
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    report_interval = n_iterations // 20  # Report progress 20 times
    last_report_time = time.time()
    
    for i in range(n_iterations):
        start_time = time.perf_counter()
        _ = likelihood_fn(test_params)
        end_time = time.perf_counter()
        
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        
        if (i + 1) % report_interval == 0:
            current_time = time.time()
            elapsed = current_time - last_report_time
            iterations_per_sec = report_interval / elapsed
            mean_time = np.mean(iteration_times)
            print(f"Completed {i + 1}/{n_iterations} iterations "
                  f"({iterations_per_sec:.1f} it/s, "
                  f"mean: {mean_time*1000:.3f} ms)")
            last_report_time = current_time
    
    profiler.disable()
    
    # Analysis
    times = np.array(iteration_times)
    print("\nFinal Timing Statistics:")
    print(f"Mean iteration time: {np.mean(times)*1000:.3f} ms")
    print(f"Median iteration time: {np.median(times)*1000:.3f} ms")
    print(f"Std deviation: {np.std(times)*1000:.3f} ms")
    print(f"Min iteration time: {np.min(times)*1000:.3f} ms")
    print(f"Max iteration time: {np.max(times)*1000:.3f} ms")
    
    # Create histogram
    plot_timing_histogram(times)
    
    # Save profiling stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.dump_stats('profile/likelihood_profile_stats.prof')
    print("\nDetailed profiling statistics saved to 'profile/likelihood_profile_stats.prof'")

if __name__ == "__main__":
    profile_likelihood() 