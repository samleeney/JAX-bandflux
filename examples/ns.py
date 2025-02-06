import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
import os
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3 import (
    optimized_salt3_multiband_flux,
)
from jax_supernovae.bandpasses import register_bandpass, get_bandpass, register_all_bandpasses
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_and_process_data
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# Settings that were previously in YAML
fix_z = True

# Nested sampling settings
NS_SETTINGS = {
    'max_iterations': int(os.environ.get('NS_MAX_ITERATIONS', '10000')),
    'n_delete': 1,
    'n_live': 125,
    'n_params': 5,
    'num_mcmc_steps_multiplier': 5
}

# Prior bounds
PRIOR_BOUNDS = {
    'z': {'min': 0.001, 'max': 0.2},
    't0': {'min': 58000.0, 'max': 59000.0},
    'x0': {'min': -5.0, 'max': -2.6},
    'x1': {'min': -4.0, 'max': 4.0},
    'c': {'min': -0.3, 'max': 0.3},
    'log_p': {'min': -4.0, 'max': -0.1}
}

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load and process data
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz', data_dir='data', fix_z=fix_z)

# Define parameter bounds and priors
if fix_z:
    param_bounds = {
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    # Create prior distributions without z
    prior_dists = {
        't0': distrax.Uniform(low=param_bounds['t0'][0], high=param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=param_bounds['x0'][0], high=param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=param_bounds['x1'][0], high=param_bounds['x1'][1]),
        'c': distrax.Uniform(low=param_bounds['c'][0], high=param_bounds['c'][1])
    }
else:
    param_bounds = {
        'z': (PRIOR_BOUNDS['z']['min'], PRIOR_BOUNDS['z']['max']),
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    # Create prior distributions with z
    prior_dists = {
        'z': distrax.Uniform(low=param_bounds['z'][0], high=param_bounds['z'][1]),
        't0': distrax.Uniform(low=param_bounds['t0'][0], high=param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=param_bounds['x0'][0], high=param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=param_bounds['x1'][0], high=param_bounds['x1'][1]),
        'c': distrax.Uniform(low=param_bounds['c'][0], high=param_bounds['c'][1])
    }

@jax.jit
def logprior(params):
    """Calculate log prior probability."""
    # Ensure params is a 2D array
    params = jnp.atleast_2d(params)
    
    if fix_z:
        # Calculate individual log probabilities without z
        logp_t0 = prior_dists['t0'].log_prob(params[:, 0])
        logp_x0 = prior_dists['x0'].log_prob(params[:, 1])
        logp_x1 = prior_dists['x1'].log_prob(params[:, 2])
        logp_c = prior_dists['c'].log_prob(params[:, 3])
        
        # Calculate total log probability
        logp = logp_t0 + logp_x0 + logp_x1 + logp_c
    else:
        # Calculate individual log probabilities with z
        logp_z = prior_dists['z'].log_prob(params[:, 0])
        logp_t0 = prior_dists['t0'].log_prob(params[:, 1])
        logp_x0 = prior_dists['x0'].log_prob(params[:, 2])
        logp_x1 = prior_dists['x1'].log_prob(params[:, 3])
        logp_c = prior_dists['c'].log_prob(params[:, 4])
        
        # Calculate total log probability
        logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c
    
    # Always return array of shape (n,)
    return jnp.reshape(logp, (-1,))

@jax.jit
def compute_single_loglikelihood(params):
    """Compute Gaussian log likelihood for a single set of parameters."""
    if fix_z:
        t0, log_x0, x1, c = params
        z = fixed_z[0]  # Use fixed redshift
    else:
        z, t0, log_x0, x1, c = params
    
    x0 = 10**(log_x0)
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    
    # Calculate model fluxes for all observations at once using optimized function
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    
    # Calculate chi-squared using JAX operations
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
    
    # Calculate log-likelihood for Gaussian distribution
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * fluxerrs**2)))
    
    return log_likelihood

@jax.jit
def compute_batch_loglikelihood(params):
    """Compute log likelihood for a batch of parameters."""
    # Ensure params is a 2D array
    params = jnp.atleast_2d(params)
    
    # Use vmap for batch processing
    batch_loglike = jax.vmap(compute_single_loglikelihood)(params)
    
    # Always return array of shape (n,)
    return jnp.reshape(batch_loglike, (-1,))

@jax.jit
def loglikelihood(params):
    """Main likelihood function for nested sampling."""
    # Ensure params is a 2D array
    params = jnp.atleast_2d(params)
    
    # Compute log-likelihoods for the batch
    batch_loglike = compute_batch_loglikelihood(params)
    
    # Always return array of shape (n,)
    return batch_loglike

def sample_from_priors(rng_key, n_samples):
    """Sample from all prior distributions at once."""
    if fix_z:
        keys = jax.random.split(rng_key, 4)
        return jnp.column_stack([
            prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
            prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
            prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
            prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,))
        ])
    else:
        keys = jax.random.split(rng_key, 5)
        return jnp.column_stack([
            prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
            prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
            prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
            prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
            prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,))
        ])

# Set up nested sampling
n_live = NS_SETTINGS['n_live']
n_params = NS_SETTINGS['n_params']
n_delete = NS_SETTINGS['n_delete']
num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']

# Initialize nested sampling algorithm
print("Setting up nested sampling algorithm...")
algo = blackjax.ns.adaptive.nss(
    logprior_fn=logprior,
    loglikelihood_fn=loglikelihood,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
)

# Initialize random key and particles
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key)

# Replace the initial particles sampling with:
initial_particles = sample_from_priors(init_key, n_live)
print("Initial particles generated, shape: ", initial_particles.shape)

# Initialize state
state = algo.init(initial_particles, compute_batch_loglikelihood)

# Define one_step function with JIT
@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

# Run nested sampling
dead = []
print("Running nested sampling...")
num_iterations = NS_SETTINGS['max_iterations']
for i in tqdm.trange(num_iterations):
    if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        break

    (state, rng_key), dead_info = one_step((state, rng_key), None)
    dead.append(dead_info)

    if i % 10 == 0:  # Print progress every 10 iterations
        print(f"Iteration {i}: logZ = {state.sampler_state.logZ:.2f}")

# Process results
dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
logw = log_weights(rng_key, dead)
logZs = jax.scipy.special.logsumexp(logw, axis=0)

print(f"Runtime evidence: {state.sampler_state.logZ:.2f}")
print(f"Estimated evidence: {logZs.mean():.2f} +- {logZs.std():.2f}")

# Save chains using the utility function
param_names = ['t0', 'log_x0', 'x1', 'c'] if fix_z else ['z', 't0', 'log_x0', 'x1', 'c']
save_chains_dead_birth(dead, param_names)

# Read the chains
samples = read_chains('chains/chains', columns=param_names)

# Create corner plot
fig, axes = make_2d_axes(param_names, figsize=(10, 10), facecolor='w')
samples.plot_2d(axes, alpha=0.9, label="posterior")
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncols=2)

plt.savefig('corner_plot.png')
plt.close()

# Print parameter statistics
print("\nParameter Statistics:")
print("-" * 50)
for param in param_names:
    mean = samples[param].mean()
    std = samples[param].std()
    print(f"{param}: {mean:.6f} ± {std:.6f}")