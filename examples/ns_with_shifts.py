"""
# Nested Sampling with JAX-bandflux including Transmission Shifts

This script demonstrates how to run the nested sampling procedure for supernovae SALT model fitting using the JAX-bandflux package with transmission shift parameters for each filter.

IMPORTANT: This example requires the Handley Lab fork of BlackJAX (not yet merged with main branch).
Install with: pip install git+https://github.com/handley-lab/blackjax@proposal
See: https://handley-lab.co.uk/nested-sampling-book/intro.html

For more examples and the complete codebase, visit the [JAX-bandflux GitHub repository](https://github.com/samleeney/JAX-bandflux).
"""

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
import os
from blackjax.ns.utils import log_weights
from jax_supernovae import SALT3Source
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_and_process_data
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# Option to fit sigma or not.
# When fit_sigma is True, the code will include an extra free parameter (log_sigma)
# so that the effective flux error becomes sigma * flux_err with sigma = 10**(log_sigma).
# When fit_sigma is False, the code will use the measured flux_err from the file.
fit_sigma = False

# Settings that were previously in YAML
fix_z = True

NS_SETTINGS = {
    'n_delete': 1,
    'n_live': 125,
    'num_mcmc_steps_multiplier': 5
}

# Prior bounds (added log_sigma bound and shift bounds)
PRIOR_BOUNDS = {
    'z': {'min': 0.001, 'max': 0.2},
    't0': {'min': 58000.0, 'max': 59000.0},
    'x0': {'min': -5.0, 'max': -2.6},
    'x1': {'min': -4.0, 'max': 4.0},
    'c': {'min': -0.3, 'max': 0.3},
    'log_sigma': {'min': -3.0, 'max': 1.0},
    'shift': {'min': -50.0, 'max': 50.0}  # ±50 Å shift bounds
}

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load and process data
times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = load_and_process_data('19dwz', data_dir='data', fix_z=fix_z)

# Create SALT3 source for bandflux calculations
source = SALT3Source()

# Get band count
n_bands = len(unique_bands)

# Define parameter bounds and priors
if fix_z:
    param_bounds = {
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    # Add shift parameters for each band
    for i in range(n_bands):
        param_bounds[f'shift_{i}'] = (PRIOR_BOUNDS['shift']['min'], PRIOR_BOUNDS['shift']['max'])
    
    if fit_sigma:
        param_bounds['log_sigma'] = (PRIOR_BOUNDS['log_sigma']['min'], PRIOR_BOUNDS['log_sigma']['max'])
    
    # Create prior distributions without z
    prior_dists = {
        't0': distrax.Uniform(low=param_bounds['t0'][0], high=param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=param_bounds['x0'][0], high=param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=param_bounds['x1'][0], high=param_bounds['x1'][1]),
        'c': distrax.Uniform(low=param_bounds['c'][0], high=param_bounds['c'][1])
    }
    # Add shift priors
    for i in range(n_bands):
        prior_dists[f'shift_{i}'] = distrax.Uniform(low=param_bounds[f'shift_{i}'][0], high=param_bounds[f'shift_{i}'][1])
    
    if fit_sigma:
        prior_dists['log_sigma'] = distrax.Uniform(low=param_bounds['log_sigma'][0], high=param_bounds['log_sigma'][1])
else:
    param_bounds = {
        'z': (PRIOR_BOUNDS['z']['min'], PRIOR_BOUNDS['z']['max']),
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    # Add shift parameters for each band
    for i in range(n_bands):
        param_bounds[f'shift_{i}'] = (PRIOR_BOUNDS['shift']['min'], PRIOR_BOUNDS['shift']['max'])
    
    if fit_sigma:
        param_bounds['log_sigma'] = (PRIOR_BOUNDS['log_sigma']['min'], PRIOR_BOUNDS['log_sigma']['max'])
    
    # Create prior distributions with z
    prior_dists = {
        'z': distrax.Uniform(low=param_bounds['z'][0], high=param_bounds['z'][1]),
        't0': distrax.Uniform(low=param_bounds['t0'][0], high=param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=param_bounds['x0'][0], high=param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=param_bounds['x1'][0], high=param_bounds['x1'][1]),
        'c': distrax.Uniform(low=param_bounds['c'][0], high=param_bounds['c'][1])
    }
    # Add shift priors
    for i in range(n_bands):
        prior_dists[f'shift_{i}'] = distrax.Uniform(low=param_bounds[f'shift_{i}'][0], high=param_bounds[f'shift_{i}'][1])
    
    if fit_sigma:
        prior_dists['log_sigma'] = distrax.Uniform(low=param_bounds['log_sigma'][0], high=param_bounds['log_sigma'][1])

@jax.jit
def logprior(params):
    """Calculate log prior probability.
    
    Parameters
    ----------
    params : array
        Parameter values to evaluate
        
    Returns
    -------
    float
        Log prior probability
    """
    if fix_z:
        idx = 0
        logp = prior_dists['t0'].log_prob(params[idx])
        idx += 1
        logp += prior_dists['x0'].log_prob(params[idx])
        idx += 1
        logp += prior_dists['x1'].log_prob(params[idx])
        idx += 1
        logp += prior_dists['c'].log_prob(params[idx])
        idx += 1
        
        # Add shift priors
        for i in range(n_bands):
            logp += prior_dists[f'shift_{i}'].log_prob(params[idx])
            idx += 1
        
        if fit_sigma:
            logp += prior_dists['log_sigma'].log_prob(params[idx])
    else:
        idx = 0
        logp = prior_dists['z'].log_prob(params[idx])
        idx += 1
        logp += prior_dists['t0'].log_prob(params[idx])
        idx += 1
        logp += prior_dists['x0'].log_prob(params[idx])
        idx += 1
        logp += prior_dists['x1'].log_prob(params[idx])
        idx += 1
        logp += prior_dists['c'].log_prob(params[idx])
        idx += 1
        
        # Add shift priors
        for i in range(n_bands):
            logp += prior_dists[f'shift_{i}'].log_prob(params[idx])
            idx += 1
        
        if fit_sigma:
            logp += prior_dists['log_sigma'].log_prob(params[idx])
    
    return logp

@jax.jit
def compute_single_loglikelihood(params):
    """Compute Gaussian log likelihood for a single set of parameters.

    Uses SALT3Source with v3.0 functional API and transmission shifts.

    Parameters
    ----------
    params : array
        Parameter values to evaluate

    Returns
    -------
    float
        Log likelihood value
    """
    # Ensure params is properly handled for both single and batched inputs
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        # If we have a batch, vmap over it
        return jax.vmap(compute_single_loglikelihood)(params)

    if fix_z:
        idx = 0
        t0 = params[idx]
        idx += 1
        log_x0 = params[idx]
        idx += 1
        x1 = params[idx]
        idx += 1
        c = params[idx]
        idx += 1

        # Extract shift parameters
        shifts = []
        for i in range(n_bands):
            shifts.append(params[idx])
            idx += 1

        if fit_sigma:
            log_sigma = params[idx]
            sigma = 10 ** log_sigma
        else:
            sigma = 1.0

        z = fixed_z[0]  # Use fixed redshift
    else:
        idx = 0
        z = params[idx]
        idx += 1
        t0 = params[idx]
        idx += 1
        log_x0 = params[idx]
        idx += 1
        x1 = params[idx]
        idx += 1
        c = params[idx]
        idx += 1

        # Extract shift parameters
        shifts = []
        for i in range(n_bands):
            shifts.append(params[idx])
            idx += 1

        if fit_sigma:
            log_sigma = params[idx]
            sigma = 10 ** log_sigma
        else:
            sigma = 1.0

    x0 = 10 ** log_x0

    # Create parameter dict for v3.0 functional API
    param_dict = {'x0': x0, 'x1': x1, 'c': c}

    # Calculate rest-frame phases from observer-frame times
    phases = (times - t0) / (1 + z)

    # Calculate model fluxes using SALT3Source with precomputed bridges and shifts
    # Note: bands parameter is not used when bridges are provided
    model_fluxes = source.bandflux(
        param_dict,
        None,  # bands not needed when using bridges
        phases,
        zp=zps,
        zpsys='ab',
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands,
        shifts=shifts
    )

    eff_fluxerrs = sigma * fluxerrs  # effective flux errors
    chi2 = jnp.sum(((fluxes - model_fluxes) / eff_fluxerrs) ** 2)
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * eff_fluxerrs ** 2)))
    return log_likelihood

def sample_from_priors(rng_key, n_samples):
    """Sample from all prior distributions at once.
    
    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        Random key for sampling
    n_samples : int
        Number of samples to generate
        
    Returns
    -------
    array
        Array of samples with shape (n_samples, n_params)
    """
    samples = []
    
    if fix_z:
        n_keys = 4 + n_bands  # t0, x0, x1, c + shifts
        if fit_sigma:
            n_keys += 1
        keys = jax.random.split(rng_key, n_keys)
        
        idx = 0
        samples.append(prior_dists['t0'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists['x0'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists['x1'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists['c'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        
        # Sample shifts
        for i in range(n_bands):
            samples.append(prior_dists[f'shift_{i}'].sample(seed=keys[idx], sample_shape=(n_samples,)))
            idx += 1
        
        if fit_sigma:
            samples.append(prior_dists['log_sigma'].sample(seed=keys[idx], sample_shape=(n_samples,)))
    else:
        n_keys = 5 + n_bands  # z, t0, x0, x1, c + shifts
        if fit_sigma:
            n_keys += 1
        keys = jax.random.split(rng_key, n_keys)
        
        idx = 0
        samples.append(prior_dists['z'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists['t0'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists['x0'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists['x1'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists['c'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        
        # Sample shifts
        for i in range(n_bands):
            samples.append(prior_dists[f'shift_{i}'].sample(seed=keys[idx], sample_shape=(n_samples,)))
            idx += 1
        
        if fit_sigma:
            samples.append(prior_dists['log_sigma'].sample(seed=keys[idx], sample_shape=(n_samples,)))
    
    return jnp.column_stack(samples)

# Adjust the total number of model parameters for nested sampling.
if fix_z:
    n_params_total = 4 + n_bands  # t0, x0, x1, c + shifts
else:
    n_params_total = 5 + n_bands  # z, t0, x0, x1, c + shifts
if fit_sigma:
    n_params_total += 1

num_mcmc_steps = n_params_total * NS_SETTINGS['num_mcmc_steps_multiplier']

# Initialize nested sampling algorithm
print("Setting up nested sampling algorithm...")
print(f"Number of bands: {n_bands}")
print(f"Total parameters: {n_params_total}")
algo = blackjax.ns.adaptive.nss(
    logprior_fn=logprior,
    loglikelihood_fn=compute_single_loglikelihood,
    n_delete=NS_SETTINGS['n_delete'],
    num_mcmc_steps=num_mcmc_steps,
)

# Initialize random key and particles
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key)

initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'])
print("Initial particles generated, shape: ", initial_particles.shape)

# Initialize state
state = algo.init(initial_particles, compute_single_loglikelihood)

# Define one_step function with JIT
@jax.jit
def one_step(carry, xs):
    """Define one step of the nested sampling algorithm.
    
    Parameters
    ----------
    carry : tuple
        (state, k) where state is the current algorithm state and k is the random key
    xs : any
        Unused placeholder for scan
        
    Returns
    -------
    tuple
        ((state, k), dead_point) where state is the updated state, k is the updated
        random key, and dead_point is the discarded point
    """
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

# Run nested sampling
dead = []
print("Running nested sampling...")
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while (not state.sampler_state.logZ_live - state.sampler_state.logZ < -3):
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(NS_SETTINGS['n_delete'])
        
        # Optional: Print progress periodically
        # if len(dead) % 10 == 0:
        #     print(f"logZ = {state.sampler_state.logZ:.2f}")

# Process results
dead = jax.tree_map(lambda *args: jnp.concatenate(args), *dead)
logw = log_weights(rng_key, dead)
logZs = jax.scipy.special.logsumexp(logw, axis=0)

print(f"Runtime evidence: {state.sampler_state.logZ:.2f}")
print(f"Estimated evidence: {logZs.mean():.2f} ± {logZs.std():.2f}")

# Save chains using the utility function
if fix_z:
    param_names = ['t0', 'log_x0', 'x1', 'c']
else:
    param_names = ['z', 't0', 'log_x0', 'x1', 'c']

# Add shift parameter names
for i in range(n_bands):
    param_names.append(f'shift_{i}')

if fit_sigma:
    param_names.append('log_sigma')

save_chains_dead_birth(dead, param_names)

# Read the chains and create visualizations
print("\\nCreating corner plot...")
samples = read_chains('chains/chains', columns=param_names)

# Create corner plot with improved styling
fig, axes = make_2d_axes(param_names, figsize=(14, 14), facecolor='w')
samples.plot_2d(axes, alpha=0.9, label="posterior")

# Improve plot aesthetics
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), 
                       loc='lower center', ncols=2)
plt.suptitle('SALT3 Parameter Posterior Distributions with Transmission Shifts', y=1.02, fontsize=14)

# Save the plot with high DPI
print("Saving corner plot to 'corner_plot_with_shifts.png'...")
plt.savefig('corner_plot_with_shifts.png', dpi=300, bbox_inches='tight')
plt.close()

# Print parameter statistics with improved formatting
print("\\nParameter Statistics:")
print("-" * 60)
print(f"{'Parameter':<15} {'Mean':>15} {'Std Dev':>15}")
print("-" * 60)
for param in param_names:
    mean = samples[param].mean()
    std = samples[param].std()
    print(f"{param:<15} {mean:>15.6f} {std:>15.6f}")
print("-" * 60)

# Optional: Save parameter statistics to a file
with open('parameter_statistics_with_shifts.txt', 'w') as f:
    f.write("Parameter Statistics:\\n")
    f.write("-" * 60 + "\\n")
    f.write(f"{'Parameter':<15} {'Mean':>15} {'Std Dev':>15}\\n")
    f.write("-" * 60 + "\\n")
    for param in param_names:
        mean = samples[param].mean()
        std = samples[param].std()
        f.write(f"{param:<15} {mean:>15.6f} {std:>15.6f}\\n")
    f.write("-" * 60 + "\\n")
print("\\nParameter statistics saved to 'parameter_statistics_with_shifts.txt'")