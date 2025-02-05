import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3nir import (
    optimized_salt3nir_multiband_flux,
)
from jax_supernovae.bandpasses import register_bandpass, get_bandpass, register_all_bandpasses
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_and_process_data
import matplotlib.pyplot as plt
import yaml
from anesthetic import read_chains, make_2d_axes


# Load settings
with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load and process data
fix_z = settings.get('fix_z', False)  # Get fix_z from settings, default False
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz', data_dir='data', fix_z=fix_z)

# Define parameter bounds and priors
if fix_z:
    # Standard parameter bounds (without log_p)
    standard_param_bounds = {
        't0': (settings['prior_bounds']['t0']['min'], settings['prior_bounds']['t0']['max']),
        'x0': (settings['prior_bounds']['x0']['min'], settings['prior_bounds']['x0']['max']),
        'x1': (settings['prior_bounds']['x1']['min'], settings['prior_bounds']['x1']['max']),
        'c': (settings['prior_bounds']['c']['min'], settings['prior_bounds']['c']['max'])
    }
    # Anomaly parameter bounds (with log_p)
    anomaly_param_bounds = {
        't0': (settings['prior_bounds']['t0']['min'], settings['prior_bounds']['t0']['max']),
        'x0': (settings['prior_bounds']['x0']['min'], settings['prior_bounds']['x0']['max']),
        'x1': (settings['prior_bounds']['x1']['min'], settings['prior_bounds']['x1']['max']),
        'c': (settings['prior_bounds']['c']['min'], settings['prior_bounds']['c']['max']),
        'log_p': (settings['prior_bounds']['log_p']['min'], settings['prior_bounds']['log_p']['max'])  # Log10 space bounds for anomaly fraction
    }
    # Create prior distributions without z for standard case
    standard_prior_dists = {
        't0': distrax.Uniform(low=standard_param_bounds['t0'][0], high=standard_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=standard_param_bounds['x0'][0], high=standard_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=standard_param_bounds['x1'][0], high=standard_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=standard_param_bounds['c'][0], high=standard_param_bounds['c'][1])
    }
    # Create prior distributions without z for anomaly case
    anomaly_prior_dists = {
        't0': distrax.Uniform(low=anomaly_param_bounds['t0'][0], high=anomaly_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=anomaly_param_bounds['x0'][0], high=anomaly_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=anomaly_param_bounds['x1'][0], high=anomaly_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=anomaly_param_bounds['c'][0], high=anomaly_param_bounds['c'][1]),
        'log_p': distrax.Uniform(low=anomaly_param_bounds['log_p'][0], high=anomaly_param_bounds['log_p'][1])
    }
else:
    # Standard parameter bounds (without log_p)
    standard_param_bounds = {
        'z': (settings['prior_bounds']['z']['min'], settings['prior_bounds']['z']['max']),
        't0': (settings['prior_bounds']['t0']['min'], settings['prior_bounds']['t0']['max']),
        'x0': (settings['prior_bounds']['x0']['min'], settings['prior_bounds']['x0']['max']),
        'x1': (settings['prior_bounds']['x1']['min'], settings['prior_bounds']['x1']['max']),
        'c': (settings['prior_bounds']['c']['min'], settings['prior_bounds']['c']['max'])
    }
    # Anomaly parameter bounds (with log_p)
    anomaly_param_bounds = {
        'z': (settings['prior_bounds']['z']['min'], settings['prior_bounds']['z']['max']),
        't0': (settings['prior_bounds']['t0']['min'], settings['prior_bounds']['t0']['max']),
        'x0': (settings['prior_bounds']['x0']['min'], settings['prior_bounds']['x0']['max']),
        'x1': (settings['prior_bounds']['x1']['min'], settings['prior_bounds']['x1']['max']),
        'c': (settings['prior_bounds']['c']['min'], settings['prior_bounds']['c']['max']),
        'log_p': (settings['prior_bounds']['log_p']['min'], settings['prior_bounds']['log_p']['max'])  # Log10 space bounds for anomaly fraction
    }
    # Create prior distributions with z for standard case
    standard_prior_dists = {
        'z': distrax.Uniform(low=standard_param_bounds['z'][0], high=standard_param_bounds['z'][1]),
        't0': distrax.Uniform(low=standard_param_bounds['t0'][0], high=standard_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=standard_param_bounds['x0'][0], high=standard_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=standard_param_bounds['x1'][0], high=standard_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=standard_param_bounds['c'][0], high=standard_param_bounds['c'][1])
    }
    # Create prior distributions with z for anomaly case
    anomaly_prior_dists = {
        'z': distrax.Uniform(low=anomaly_param_bounds['z'][0], high=anomaly_param_bounds['z'][1]),
        't0': distrax.Uniform(low=anomaly_param_bounds['t0'][0], high=anomaly_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=anomaly_param_bounds['x0'][0], high=anomaly_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=anomaly_param_bounds['x1'][0], high=anomaly_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=anomaly_param_bounds['c'][0], high=anomaly_param_bounds['c'][1]),
        'log_p': distrax.Uniform(low=anomaly_param_bounds['log_p'][0], high=anomaly_param_bounds['log_p'][1])
    }

@jax.jit
def logprior(params):
    """Calculate log prior probability."""
    # Ensure params is a 2D array
    params = jnp.atleast_2d(params)
    
    if fix_z:
        if params.shape[1] == 4:  # Standard case
            # Calculate individual log probabilities without z
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 2])
            logp_c = standard_prior_dists['c'].log_prob(params[:, 3])
            
            # Calculate total log probability
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c
        else:  # Anomaly case
            # Calculate individual log probabilities without z
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 2])
            logp_c = anomaly_prior_dists['c'].log_prob(params[:, 3])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 4])
            
            # Calculate total log probability
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_logp
    else:
        if params.shape[1] == 5:  # Standard case
            # Calculate individual log probabilities with z
            logp_z = standard_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 3])
            logp_c = standard_prior_dists['c'].log_prob(params[:, 4])
            
            # Calculate total log probability
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c
        else:  # Anomaly case
            # Calculate individual log probabilities with z
            logp_z = anomaly_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 3])
            logp_c = anomaly_prior_dists['c'].log_prob(params[:, 4])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 5])
            
            # Calculate total log probability
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_logp
    
    # Always return array of shape (n,)
    return jnp.reshape(logp, (-1,))

@jax.jit
def compute_single_loglikelihood_standard(params):
    """Compute standard Gaussian log likelihood without anomaly detection."""
    if fix_z:
        t0, log_x0, x1, c = params
        z = fixed_z[0]  # Use fixed redshift
    else:
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
def compute_batch_loglikelihood_standard(params):
    """Compute standard log likelihood for a batch of parameters."""
    params = jnp.atleast_2d(params)
    batch_loglike = jax.vmap(compute_single_loglikelihood_standard)(params)
    return jnp.reshape(batch_loglike, (-1,))

@jax.jit
def loglikelihood_standard(params):
    """Main standard likelihood function for nested sampling."""
    params = jnp.atleast_2d(params)
    batch_loglike = compute_batch_loglikelihood_standard(params)
    return batch_loglike

@jax.jit
def compute_single_loglikelihood(params):
    """Compute Gaussian log likelihood for a single set of parameters."""
    if fix_z:
        t0, log_x0, x1, c, log_p = params
        z = fixed_z[0]  # Use fixed redshift
    else:
        z, t0, log_x0, x1, c, log_p = params
    
    x0 = 10**(log_x0)
    p = 10**(log_p)  # Transform log10_p to p
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    
    # Calculate model fluxes for all observations at once using optimized function
    model_fluxes = optimized_salt3nir_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    
    # Calculate individual point log likelihoods
    point_logL = -0.5 * ((fluxes - model_fluxes) / fluxerrs)**2 - 0.5 * jnp.log(2 * jnp.pi * fluxerrs**2) + jnp.log(1 - p)
    
    # Calculate threshold for anomalies
    delta = 250  # ~max data
    
    # Apply anomaly detection
    emax = point_logL > (log_p - jnp.log(delta))
    logL = jnp.where(emax, point_logL, log_p - jnp.log(delta))
    
    # Sum the log likelihoods
    total_logL = jnp.sum(logL)
    
    return total_logL

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

def sample_from_priors(rng_key, n_samples, loglikelihood_fn):
    """Sample from all prior distributions at once."""
    if fix_z:
        if loglikelihood_fn == loglikelihood_standard:  # Standard case
            keys = jax.random.split(rng_key, 4)
            return jnp.column_stack([
                standard_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                standard_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                standard_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                standard_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,))
            ])
        else:  # Anomaly case
            keys = jax.random.split(rng_key, 5)
            return jnp.column_stack([
                anomaly_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                anomaly_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                anomaly_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                anomaly_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                anomaly_prior_dists['log_p'].sample(seed=keys[4], sample_shape=(n_samples,))
            ])
    else:
        if loglikelihood_fn == loglikelihood_standard:  # Standard case
            keys = jax.random.split(rng_key, 5)
            return jnp.column_stack([
                standard_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                standard_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                standard_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                standard_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                standard_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,))
            ])
        else:  # Anomaly case
            keys = jax.random.split(rng_key, 6)
            return jnp.column_stack([
                anomaly_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                anomaly_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                anomaly_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                anomaly_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                anomaly_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                anomaly_prior_dists['log_p'].sample(seed=keys[5], sample_shape=(n_samples,))
            ])

def run_nested_sampling(loglikelihood_fn, output_prefix, num_iterations=settings['nested_sampling']['max_iterations']):
    """Run nested sampling with given likelihood function and save results."""
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_prefix, exist_ok=True)
    
    print("Setting up nested sampling algorithm...")
    # Initialize nested sampling algorithm
    algo = blackjax.ns.adaptive.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglikelihood_fn,
        n_delete=n_delete,
        num_mcmc_steps=num_mcmc_steps,
    )

    # Initialize random key and particles
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = sample_from_priors(init_key, n_live, loglikelihood_fn)
    print("Initial particles generated, shape: ", initial_particles.shape)
    
    # Initialize state with the correct likelihood function
    if loglikelihood_fn == loglikelihood_standard:
        state = algo.init(initial_particles, compute_batch_loglikelihood_standard)
    else:
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
    print(f"Running nested sampling for {output_prefix}...")
    for i in tqdm.trange(settings['nested_sampling']['max_iterations']):
        if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
            break

        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        
        if i % 10 == 0:
            print(f"Iteration {i}: logZ = {state.sampler_state.logZ:.2f}")
    
    # Process results
    dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
    logw = log_weights(rng_key, dead)
    logZs = jax.scipy.special.logsumexp(logw, axis=0)
    
    print(f"Runtime evidence: {state.sampler_state.logZ:.2f}")
    print(f"Estimated evidence: {logZs.mean():.2f} +- {logZs.std():.2f}")
    
    # Save chains with full path
    if loglikelihood_fn == loglikelihood_standard:
        param_names = ['t0', 'log_x0', 'x1', 'c'] if fix_z else ['z', 't0', 'log_x0', 'x1', 'c']
    else:
        param_names = ['t0', 'log_x0', 'x1', 'c', 'log_p'] if fix_z else ['z', 't0', 'log_x0', 'x1', 'c', 'log_p']
    save_chains_dead_birth(dead, param_names, root_dir=output_prefix)

# Set up nested sampling
n_live = settings['nested_sampling']['n_live']
n_delete = settings['nested_sampling']['n_delete']

def get_n_params(loglikelihood_fn):
    """Get number of parameters based on likelihood function and fix_z setting."""
    if loglikelihood_fn == loglikelihood_standard:
        return 4 if fix_z else 5
    else:
        return 5 if fix_z else 6

if __name__ == "__main__":
    # Run standard version
    print("Running standard version...")
    n_params = get_n_params(loglikelihood_standard)
    num_mcmc_steps = n_params * settings['nested_sampling']['num_mcmc_steps_multiplier']
    standard_samples = run_nested_sampling(loglikelihood_standard, "chains_standard")
    
    # Run anomaly version
    print("\nRunning anomaly detection version...")
    n_params = get_n_params(loglikelihood)
    num_mcmc_steps = n_params * settings['nested_sampling']['num_mcmc_steps_multiplier']
    anomaly_samples = run_nested_sampling(loglikelihood, "chains_anomaly")