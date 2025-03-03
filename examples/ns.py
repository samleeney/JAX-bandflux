
'''
# Nested Sampling with JAX-bandflux

This script demonstrates how to run the nested sampling procedure for supernovae SALT model fitting using the JAX-bandflux package. We will install the package, load the data, set up and run the nested sampling algorithm, and finally produce a corner plot of the posterior samples.

For more examples and the complete codebase, visit the [JAX-bandflux GitHub repository](https://github.com/samleeney/JAX-bandflux). The academic paper associated with this work can be found [here](https://github.com/samleeney/JAX-bandflux/blob/71ca8d1b3b273147e1e9bf60a9ef11a806363b80/paper.bib).
'''


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

# Option to fit sigma or not.
# When fit_sigma is True, the code will include an extra free parameter (log_sigma)
# so that the effective flux error becomes sigma * flux_err with sigma = 10**(log_sigma).
# When fit_sigma is False, the code will use the measured flux_err from the file.
fit_sigma = False

# Settings that were previously in YAML
fix_z = True

NS_SETTINGS = {
    'max_iterations': int(os.environ.get('NS_MAX_ITERATIONS', '5000')),
    'n_delete': 1,
    'n_live': 125,
    'num_mcmc_steps_multiplier': 5
}

# Prior bounds (added log_sigma bound)
PRIOR_BOUNDS = {
    'z': {'min': 0.001, 'max': 0.2},
    't0': {'min': 58000.0, 'max': 59000.0},
    'x0': {'min': -5.0, 'max': -2.6},
    'x1': {'min': -4.0, 'max': 4.0},
    'c': {'min': -0.3, 'max': 0.3},
    'log_sigma': {'min': -3.0, 'max': 1.0}
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
    if fit_sigma:
        param_bounds['log_sigma'] = (PRIOR_BOUNDS['log_sigma']['min'], PRIOR_BOUNDS['log_sigma']['max'])
    # Create prior distributions without z
    prior_dists = {
        't0': distrax.Uniform(low=param_bounds['t0'][0], high=param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=param_bounds['x0'][0], high=param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=param_bounds['x1'][0], high=param_bounds['x1'][1]),
        'c': distrax.Uniform(low=param_bounds['c'][0], high=param_bounds['c'][1])
    }
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
    if fit_sigma:
        prior_dists['log_sigma'] = distrax.Uniform(low=param_bounds['log_sigma'][0], high=param_bounds['log_sigma'][1])

@jax.jit
def logprior(params):
    """Calculate log prior probability."""
    params = jnp.atleast_2d(params)
    if fix_z:
        if fit_sigma:
            logp_t0 = prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = prior_dists['x1'].log_prob(params[:, 2])
            logp_c = prior_dists['c'].log_prob(params[:, 3])
            logp_sigma = prior_dists['log_sigma'].log_prob(params[:, 4])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma
        else:
            logp_t0 = prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = prior_dists['x1'].log_prob(params[:, 2])
            logp_c = prior_dists['c'].log_prob(params[:, 3])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c
    else:
        if fit_sigma:
            logp_z = prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = prior_dists['x1'].log_prob(params[:, 3])
            logp_c = prior_dists['c'].log_prob(params[:, 4])
            logp_sigma = prior_dists['log_sigma'].log_prob(params[:, 5])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma
        else:
            logp_z = prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = prior_dists['x1'].log_prob(params[:, 3])
            logp_c = prior_dists['c'].log_prob(params[:, 4])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c
    return jnp.reshape(logp, (-1,))

@jax.jit
def compute_single_loglikelihood(params):
    """Compute Gaussian log likelihood for a single set of parameters."""
    if fix_z:
        if fit_sigma:
            t0, log_x0, x1, c, log_sigma = params
            sigma = 10 ** log_sigma
        else:
            t0, log_x0, x1, c = params
            sigma = 1.0
        z = fixed_z[0]  # Use fixed redshift
    else:
        if fit_sigma:
            z, t0, log_x0, x1, c, log_sigma = params
            sigma = 10 ** log_sigma
        else:
            z, t0, log_x0, x1, c = params
            sigma = 1.0
    x0 = 10 ** log_x0
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}

    # Calculate model fluxes for all observations at once using optimized function
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]

    eff_fluxerrs = sigma * fluxerrs  # effective flux errors
    chi2 = jnp.sum(((fluxes - model_fluxes) / eff_fluxerrs) ** 2)
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * eff_fluxerrs ** 2)))
    return log_likelihood

@jax.jit
def compute_batch_loglikelihood(params):
    """Compute log likelihood for a batch of parameters."""
    params = jnp.atleast_2d(params)
    batch_loglike = jax.vmap(compute_single_loglikelihood)(params)
    return jnp.reshape(batch_loglike, (-1,))

@jax.jit
def loglikelihood(params):
    """Main likelihood function for nested sampling."""
    params = jnp.atleast_2d(params)
    batch_loglike = compute_batch_loglikelihood(params)
    return batch_loglike

def sample_from_priors(rng_key, n_samples):
    """Sample from all prior distributions at once."""
    if fix_z:
        if fit_sigma:
            keys = jax.random.split(rng_key, 5)
            return jnp.column_stack([
                prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                prior_dists['log_sigma'].sample(seed=keys[4], sample_shape=(n_samples,))
            ])
        else:
            keys = jax.random.split(rng_key, 4)
            return jnp.column_stack([
                prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,))
            ])
    else:
        if fit_sigma:
            keys = jax.random.split(rng_key, 6)
            return jnp.column_stack([
                prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                prior_dists['log_sigma'].sample(seed=keys[5], sample_shape=(n_samples,))
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

# Adjust the total number of model parameters for nested sampling.
if fix_z:
    n_params_total = 4
else:
    n_params_total = 5
if fit_sigma:
    n_params_total += 1

num_mcmc_steps = n_params_total * NS_SETTINGS['num_mcmc_steps_multiplier']

# Initialize nested sampling algorithm
print("Setting up nested sampling algorithm...")
algo = blackjax.ns.adaptive.nss(
    logprior_fn=logprior,
    loglikelihood_fn=loglikelihood,
    n_delete=NS_SETTINGS['n_delete'],
    num_mcmc_steps=num_mcmc_steps,
)

# Initialize random key and particles
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key)

initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'])
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
if fix_z:
    param_names = ['t0', 'log_x0', 'x1', 'c']
else:
    param_names = ['z', 't0', 'log_x0', 'x1', 'c']
if fit_sigma:
    param_names.append('log_sigma')
save_chains_dead_birth(dead, param_names)

# Read the chains and create visualizations
print("\nCreating corner plot...")
samples = read_chains('chains/chains', columns=param_names)

# Create corner plot with improved styling
fig, axes = make_2d_axes(param_names, figsize=(12, 12), facecolor='w')
samples.plot_2d(axes, alpha=0.9, label="posterior")

# Improve plot aesthetics
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), 
                       loc='lower center', ncols=2)
plt.suptitle('SALT3 Parameter Posterior Distributions', y=1.02, fontsize=14)

# Save the plot with high DPI
print("Saving corner plot to 'corner_plot.png'...")
plt.savefig('corner_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Print parameter statistics with improved formatting
print("\nParameter Statistics:")
print("-" * 50)
print(f"{'Parameter':<12} {'Mean':>15} {'Std Dev':>15}")
print("-" * 50)
for param in param_names:
    mean = samples[param].mean()
    std = samples[param].std()
    print(f"{param:<12} {mean:>15.6f} {std:>15.6f}")
print("-" * 50)

# Optional: Save parameter statistics to a file
with open('parameter_statistics.txt', 'w') as f:
    f.write("Parameter Statistics:\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Parameter':<12} {'Mean':>15} {'Std Dev':>15}\n")
    f.write("-" * 50 + "\n")
    for param in param_names:
        mean = samples[param].mean()
        std = samples[param].std()
        f.write(f"{param:<12} {mean:>15.6f} {std:>15.6f}\n")
    f.write("-" * 50 + "\n")
print("\nParameter statistics saved to 'parameter_statistics.txt'")