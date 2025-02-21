import distrax
import jax
import jax.numpy as jnp
import numpy as np
import time
import cProfile
import pstats
from functools import partial
import blackjax
from examples.ns import (
    logprior,
    loglikelihood,
    compute_batch_loglikelihood,
    sample_from_priors,
    fit_sigma,
    fix_z
)

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Define minimal settings for profiling
PROFILE_SETTINGS = {
    'n_delete': 1,
    'n_live': 125,
    'num_mcmc_steps_multiplier': 5
}

def setup_ns():
    """Set up nested sampling algorithm and initial state."""
    # Adjust parameters based on fix_z and fit_sigma
    if fix_z:
        n_params_total = 4
    else:
        n_params_total = 5
    if fit_sigma:
        n_params_total += 1

    num_mcmc_steps = n_params_total * PROFILE_SETTINGS['num_mcmc_steps_multiplier']

    # Initialize nested sampling algorithm
    algo = blackjax.ns.adaptive.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglikelihood,
        n_delete=PROFILE_SETTINGS['n_delete'],
        num_mcmc_steps=num_mcmc_steps,
    )

    # Initialize random key and particles
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)

    initial_particles = sample_from_priors(init_key, PROFILE_SETTINGS['n_live'])
    
    # Initialize state without running the main loop
    state = algo.init(initial_particles, compute_batch_loglikelihood)
    
    return algo, state, rng_key

@partial(jax.jit, static_argnums=(0,))
def one_step(algo, state, key):
    """Single nested sampling step."""
    key, subkey = jax.random.split(key)
    state, dead_point = algo.step(subkey, state)
    return state, key, dead_point

def profile_ns_iterations(n_iterations=100, warmup=5):
    """Profile nested sampling iterations."""
    print(f"Profiling {n_iterations} nested sampling iterations (ignoring first {warmup} iterations)")
    
    # Setup
    print("\nInitializing nested sampling...")
    algo, state, rng_key = setup_ns()
    
    # Warmup phase
    print("\nPerforming warmup iterations...")
    for i in range(warmup):
        print(f"Warmup iteration {i+1}/{warmup}")
        state, rng_key, _ = one_step(algo, state, rng_key)
    
    # Profiling phase
    print("\nStarting profiling...")
    iteration_times = []
    
    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    for i in range(n_iterations):
        start_time = time.perf_counter()
        state, rng_key, _ = one_step(algo, state, rng_key)
        end_time = time.perf_counter()
        
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_iterations} iterations")
            print(f"Current mean iteration time: {np.mean(iteration_times):.6f} seconds")
    
    profiler.disable()
    
    # Analysis
    times = np.array(iteration_times)
    print("\nFinal Timing Statistics:")
    print(f"Mean iteration time: {np.mean(times):.6f} seconds")
    print(f"Std deviation: {np.std(times):.6f} seconds")
    print(f"Min iteration time: {np.min(times):.6f} seconds")
    print(f"Max iteration time: {np.max(times):.6f} seconds")
    
    # Save detailed profiling stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.dump_stats('ns_profile_stats.prof')
    print("\nDetailed profiling statistics saved to 'ns_profile_stats.prof'")

if __name__ == "__main__":
    profile_ns_iterations(n_iterations=100, warmup=5) 