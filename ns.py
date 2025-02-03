import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3nir import (
    optimized_salt3nir_multiband_flux,
    optimized_salt3nir_bandflux,
    precompute_bandflux_bridge
)
from jax_supernovae.bandpasses import register_bandpass, get_bandpass, register_all_bandpasses
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_and_process_data
import matplotlib.pyplot as plt
import yaml

# Load settings
with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load and process data
times, fluxes, fluxerrs, zps, band_indices, bridges = load_and_process_data('19agl')

# Define parameter bounds and priors
param_bounds = {
    'z': (settings['prior_bounds']['z']['min'], settings['prior_bounds']['z']['max']),
    't0': (settings['prior_bounds']['t0']['min'], settings['prior_bounds']['t0']['max']),
    'x0': (settings['prior_bounds']['x0']['min'], settings['prior_bounds']['x0']['max']),
    'x1': (settings['prior_bounds']['x1']['min'], settings['prior_bounds']['x1']['max']),
    'c': (settings['prior_bounds']['c']['min'], settings['prior_bounds']['c']['max'])
}

# Create prior distributions
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
    
    # Calculate individual log probabilities
    logp_z = prior_dists['z'].log_prob(params[:, 0])
    logp_t0 = prior_dists['t0'].log_prob(params[:, 1])
    logp_x0 = prior_dists['x0'].log_prob(params[:, 2])
    logp_x1 = prior_dists['x1'].log_prob(params[:, 3])
    logp_c = prior_dists['c'].log_prob(params[:, 4])
    
    # Calculate total log probability
    logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c
    
    # Always return array of shape (n,)
    return jnp.reshape(logp, (-1,))

def sample_from_priors(rng_key, n_samples):
    """Sample from all prior distributions at once."""
    keys = jax.random.split(rng_key, 5)
    return jnp.column_stack([
        prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
        prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
        prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
        prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
        prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,))
    ])

@jax.jit
def compute_single_loglikelihood(params):
    """Compute Gaussian log likelihood for a single set of parameters."""
    z, t0, log_x0, x1, c = params
    x0 = 10**(log_x0)
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    
    # Calculate model fluxes for all observations at once using optimized function
    model_fluxes = optimized_salt3nir_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
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

# Set up nested sampling
n_live = settings['nested_sampling']['n_live']
n_params = settings['nested_sampling']['n_params']
n_delete = settings['nested_sampling']['n_delete']
num_mcmc_steps = n_params * settings['nested_sampling']['num_mcmc_steps_multiplier']

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
num_iterations = settings['nested_sampling']['max_iterations']
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
param_names = ['z', 't0', 'x0', 'x1', 'c']
save_chains_dead_birth(dead, param_names) 