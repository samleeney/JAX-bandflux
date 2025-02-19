import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
import os
import yaml
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.bandpasses import register_bandpass, get_bandpass, register_all_bandpasses
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_and_process_data
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# Define default settings for nested sampling and prior bounds
DEFAULT_NS_SETTINGS = {
    'max_iterations': int(os.environ.get('NS_MAX_ITERATIONS', '50')),
    'n_delete': 1,
    'n_live': 200,
    'num_mcmc_steps_multiplier': 10,
    'fit_sigma': False,
    'fit_log_p': True,
    'fit_z': True
}

DEFAULT_PRIOR_BOUNDS = {
    'z': {'min': 0.001, 'max': 0.2},
    't0': {'min': 58000.0, 'max': 60000.0},
    'x0': {'min': -5.0, 'max': -1},
    'x1': {'min': -4.0, 'max': 4.0},
    'c': {'min': -0.4, 'max': 0.4},
    'sigma': {'min': 0.001, 'max': 5},
    'log_p': {'min': -20, 'max': -2}
}

# Default settings
DEFAULT_SETTINGS = {
    'fix_z': True,
    'sn_name': '21yrf'  # Default supernova to analyze
}

# Try to load settings.yaml; if not found, use an empty dictionary
try:
    with open('settings.yaml', 'r') as f:
        settings_from_file = yaml.safe_load(f)
except FileNotFoundError:
    settings_from_file = {}

# Merge the settings from file with the defaults
settings = DEFAULT_SETTINGS.copy()
settings.update(settings_from_file)

fix_z = settings['fix_z']
sn_name = settings['sn_name']

NS_SETTINGS = DEFAULT_NS_SETTINGS.copy()
NS_SETTINGS.update(settings.get('nested_sampling', {}))

PRIOR_BOUNDS = DEFAULT_PRIOR_BOUNDS.copy()
if 'prior_bounds' in settings:
    PRIOR_BOUNDS.update(settings['prior_bounds'])

# Option flag: when fit_sigma is True, an extra parameter is added
fit_sigma = NS_SETTINGS['fit_sigma']

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load and process data
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(sn_name, data_dir='jax_supernovae/data', fix_z=fix_z)

# =============================================================================
# Set up parameter bounds and prior distributions for the standard (non‐anomaly)
# nested sampling version.
# =============================================================================
if fix_z:
    standard_param_bounds = {
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    if fit_sigma:
        standard_param_bounds['sigma'] = (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])
else:
    standard_param_bounds = {
        'z': (PRIOR_BOUNDS['z']['min'], PRIOR_BOUNDS['z']['max']),
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    if fit_sigma:
        standard_param_bounds['sigma'] = (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])

if fix_z:
    standard_prior_dists = {
        't0': distrax.Uniform(low=standard_param_bounds['t0'][0], high=standard_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=standard_param_bounds['x0'][0], high=standard_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=standard_param_bounds['x1'][0], high=standard_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=standard_param_bounds['c'][0], high=standard_param_bounds['c'][1])
    }
    if fit_sigma:
        standard_prior_dists['sigma'] = distrax.Uniform(low=standard_param_bounds['sigma'][0], high=standard_param_bounds['sigma'][1])
else:
    standard_prior_dists = {
        'z': distrax.Uniform(low=standard_param_bounds['z'][0], high=standard_param_bounds['z'][1]),
        't0': distrax.Uniform(low=standard_param_bounds['t0'][0], high=standard_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=standard_param_bounds['x0'][0], high=standard_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=standard_param_bounds['x1'][0], high=standard_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=standard_param_bounds['c'][0], high=standard_param_bounds['c'][1])
    }
    if fit_sigma:
        standard_prior_dists['sigma'] = distrax.Uniform(low=standard_param_bounds['sigma'][0], high=standard_param_bounds['sigma'][1])

# =============================================================================
# Set up parameter bounds and priors for the anomaly detection version.
# An extra parameter 'log_p' is included.
# =============================================================================
if fix_z:
    anomaly_param_bounds = {
        't0': (standard_param_bounds['t0'][0], standard_param_bounds['t0'][1]),
        'x0': (standard_param_bounds['x0'][0], standard_param_bounds['x0'][1]),
        'x1': (standard_param_bounds['x1'][0], standard_param_bounds['x1'][1]),
        'c': (standard_param_bounds['c'][0], standard_param_bounds['c'][1]),
        'log_p': (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    }
    if fit_sigma:
        anomaly_param_bounds['sigma'] = (standard_param_bounds['sigma'][0], standard_param_bounds['sigma'][1])
else:
    anomaly_param_bounds = {
        'z': (standard_param_bounds['z'][0], standard_param_bounds['z'][1]),
        't0': (standard_param_bounds['t0'][0], standard_param_bounds['t0'][1]),
        'x0': (standard_param_bounds['x0'][0], standard_param_bounds['x0'][1]),
        'x1': (standard_param_bounds['x1'][0], standard_param_bounds['x1'][1]),
        'c': (standard_param_bounds['c'][0], standard_param_bounds['c'][1]),
        'log_p': (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    }
    if fit_sigma:
        anomaly_param_bounds['sigma'] = (standard_param_bounds['sigma'][0], standard_param_bounds['sigma'][1])

if fix_z:
    anomaly_prior_dists = {
        't0': distrax.Uniform(low=anomaly_param_bounds['t0'][0], high=anomaly_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=anomaly_param_bounds['x0'][0], high=anomaly_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=anomaly_param_bounds['x1'][0], high=anomaly_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=anomaly_param_bounds['c'][0], high=anomaly_param_bounds['c'][1]),
        'log_p': distrax.Uniform(low=anomaly_param_bounds['log_p'][0], high=anomaly_param_bounds['log_p'][1])
    }
    if fit_sigma:
        anomaly_prior_dists['sigma'] = distrax.Uniform(low=anomaly_param_bounds['sigma'][0], high=anomaly_param_bounds['sigma'][1])
else:
    anomaly_prior_dists = {
        'z': distrax.Uniform(low=anomaly_param_bounds['z'][0], high=anomaly_param_bounds['z'][1]),
        't0': distrax.Uniform(low=anomaly_param_bounds['t0'][0], high=anomaly_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=anomaly_param_bounds['x0'][0], high=anomaly_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=anomaly_param_bounds['x1'][0], high=anomaly_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=anomaly_param_bounds['c'][0], high=anomaly_param_bounds['c'][1]),
        'log_p': distrax.Uniform(low=anomaly_param_bounds['log_p'][0], high=anomaly_param_bounds['log_p'][1])
    }
    if fit_sigma:
        anomaly_prior_dists['sigma'] = distrax.Uniform(low=anomaly_param_bounds['sigma'][0], high=anomaly_param_bounds['sigma'][1])

# =============================================================================
# Standard likelihood functions (using salt3 multiband flux).
# =============================================================================
@jax.jit
def logprior_standard(params):
    """Calculate log prior probability for standard nested sampling."""
    params = jnp.atleast_2d(params)
    if fix_z:
        if fit_sigma:
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 2])
            logp_c  = standard_prior_dists['c'].log_prob(params[:, 3])
            logp_sigma = standard_prior_dists['sigma'].log_prob(params[:, 4])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma
        else:
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 2])
            logp_c  = standard_prior_dists['c'].log_prob(params[:, 3])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c
    else:
        if fit_sigma:
            logp_z  = standard_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 3])
            logp_c  = standard_prior_dists['c'].log_prob(params[:, 4])
            logp_sigma = standard_prior_dists['sigma'].log_prob(params[:, 5])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma
        else:
            logp_z  = standard_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 3])
            logp_c  = standard_prior_dists['c'].log_prob(params[:, 4])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c
    return jnp.reshape(logp, (-1,))

@jax.jit
def logprior_anomaly(params):
    """Calculate log prior probability for anomaly detection nested sampling."""
    params = jnp.atleast_2d(params)
    if fix_z:
        if fit_sigma:
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 2])
            logp_c  = anomaly_prior_dists['c'].log_prob(params[:, 3])
            logp_sigma = anomaly_prior_dists['sigma'].log_prob(params[:, 4])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 5])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma + logp_logp
        else:
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 2])
            logp_c  = anomaly_prior_dists['c'].log_prob(params[:, 3])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 4])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_logp
    else:
        if fit_sigma:
            logp_z  = anomaly_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 3])
            logp_c  = anomaly_prior_dists['c'].log_prob(params[:, 4])
            logp_sigma = anomaly_prior_dists['sigma'].log_prob(params[:, 5])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 6])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma + logp_logp
        else:
            logp_z  = anomaly_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 3])
            logp_c  = anomaly_prior_dists['c'].log_prob(params[:, 4])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 5])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_logp
    return jnp.reshape(logp, (-1,))

@jax.jit
def compute_single_loglikelihood_standard(params):
    """Compute Gaussian log likelihood for a single set of parameters (standard)."""
    if fix_z:
        if fit_sigma:
            t0, log_x0, x1, c, sigma = params
        else:
            t0, log_x0, x1, c = params
            sigma = 1.0  # Default value when not fitting sigma
        z = fixed_z[0]
    else:
        if fit_sigma:
            z, t0, log_x0, x1, c, sigma = params
        else:
            z, t0, log_x0, x1, c = params
            sigma = 1.0  # Default value when not fitting sigma
    x0 = 10 ** log_x0
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    eff_fluxerrs = sigma * fluxerrs
    chi2 = jnp.sum(((fluxes - model_fluxes) / eff_fluxerrs) ** 2)
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * eff_fluxerrs ** 2)))
    return log_likelihood

@jax.jit
def compute_batch_loglikelihood_standard(params):
    params = jnp.atleast_2d(params)
    batch_loglike = jax.vmap(compute_single_loglikelihood_standard)(params)
    return jnp.reshape(batch_loglike, (-1,))

@jax.jit
def loglikelihood_standard(params):
    params = jnp.atleast_2d(params)
    batch_loglike = compute_batch_loglikelihood_standard(params)
    return batch_loglike

# =============================================================================
# Anomaly detection likelihood functions (using salt3 multiband flux).
# An extra parameter 'log_p' is used to weight the likelihood for anomalies.
# =============================================================================
@jax.jit
def compute_single_loglikelihood_anomaly(params):
    """Compute Gaussian log likelihood for a single set of parameters with anomaly detection."""
    if fix_z:
        if fit_sigma:
            t0, log_x0, x1, c, sigma, log_p = params
        else:
            t0, log_x0, x1, c, log_p = params
            sigma = 1.0  # Default value when not fitting sigma
        z = fixed_z[0]
    else:
        if fit_sigma:
            z, t0, log_x0, x1, c, sigma, log_p = params
        else:
            z, t0, log_x0, x1, c, log_p = params
            sigma = 1.0  # Default value when not fitting sigma
    x0 = 10 ** log_x0
    p = jnp.exp(log_p)  # Changed: Now using natural exponential
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    eff_fluxerrs = sigma * fluxerrs
    point_logL = -0.5 * (((fluxes - model_fluxes) / eff_fluxerrs) ** 2) - 0.5 * jnp.log(2 * jnp.pi * eff_fluxerrs ** 2) + jnp.log(1 - p)
    delta = jnp.max(jnp.abs(fluxes))  # Use maximum absolute flux value as delta
    emax = point_logL > (log_p - jnp.log(delta))  # Now consistent as both are natural logs
    logL = jnp.where(emax, point_logL, log_p - jnp.log(delta))
    total_logL = jnp.sum(logL)
    return total_logL, emax

@jax.jit
def compute_batch_loglikelihood_anomaly(params):
    params = jnp.atleast_2d(params)
    batch_loglike, batch_emax = jax.vmap(compute_single_loglikelihood_anomaly)(params)
    return jnp.reshape(batch_loglike, (-1,)), batch_emax

@jax.jit
def loglikelihood_anomaly(params):
    params = jnp.atleast_2d(params)
    batch_loglike, batch_emax = compute_batch_loglikelihood_anomaly(params)
    return batch_loglike

# =============================================================================
# Function to sample from the prior distributions.
# It chooses between the standard and anomaly priors based on the likelihood function.
# =============================================================================
def sample_from_priors(rng_key, n_samples, ll_fn=loglikelihood_standard):
    if ll_fn == loglikelihood_anomaly:
        if fix_z:
            if fit_sigma:
                keys = jax.random.split(rng_key, 6)
                return jnp.column_stack([
                    anomaly_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    anomaly_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    anomaly_prior_dists['sigma'].sample(seed=keys[4], sample_shape=(n_samples,)),
                    anomaly_prior_dists['log_p'].sample(seed=keys[5], sample_shape=(n_samples,))
                ])
            else:
                keys = jax.random.split(rng_key, 5)
                return jnp.column_stack([
                    anomaly_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    anomaly_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    anomaly_prior_dists['log_p'].sample(seed=keys[4], sample_shape=(n_samples,))
                ])
        else:
            if fit_sigma:
                keys = jax.random.split(rng_key, 7)
                return jnp.column_stack([
                    anomaly_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    anomaly_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    anomaly_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                    anomaly_prior_dists['sigma'].sample(seed=keys[5], sample_shape=(n_samples,)),
                    anomaly_prior_dists['log_p'].sample(seed=keys[6], sample_shape=(n_samples,))
                ])
            else:
                keys = jax.random.split(rng_key, 6)
                return jnp.column_stack([
                    anomaly_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    anomaly_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    anomaly_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                    anomaly_prior_dists['log_p'].sample(seed=keys[5], sample_shape=(n_samples,))
                ])
    else:  # Standard case
        if fix_z:
            if fit_sigma:
                keys = jax.random.split(rng_key, 5)
                return jnp.column_stack([
                    standard_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    standard_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    standard_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    standard_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    standard_prior_dists['sigma'].sample(seed=keys[4], sample_shape=(n_samples,))
                ])
            else:
                keys = jax.random.split(rng_key, 4)
                return jnp.column_stack([
                    standard_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    standard_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    standard_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    standard_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,))
                ])
        else:
            if fit_sigma:
                keys = jax.random.split(rng_key, 6)
                return jnp.column_stack([
                    standard_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    standard_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    standard_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    standard_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    standard_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                    standard_prior_dists['sigma'].sample(seed=keys[5], sample_shape=(n_samples,))
                ])
            else:
                keys = jax.random.split(rng_key, 5)
                return jnp.column_stack([
                    standard_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    standard_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    standard_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    standard_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    standard_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,))
                ])

# =============================================================================
# Set the total number of model parameters for the standard case.
# =============================================================================
n_params_total = 5 if fix_z else 6
num_mcmc_steps = n_params_total * NS_SETTINGS['num_mcmc_steps_multiplier']

# =============================================================================
# Function to run nested sampling.
# It initialises the BlackJAX nested sampler, runs the sampling loop,
# and saves output chains (and weighted anomaly indicators for the anomaly run).
# =============================================================================
def run_nested_sampling(ll_fn, output_prefix, sn_name, identifier="", num_iterations=NS_SETTINGS['max_iterations']):
    """Run nested sampling with output directories/files including supernova name.
    
    Args:
        ll_fn: Likelihood function to use
        output_prefix: Base prefix for output directory ('chains_standard' or 'chains_anomaly')
        sn_name: Name of the supernova (e.g. '20aai')
        identifier: Additional string to append to output directory and filenames
        num_iterations: Maximum number of iterations
    """
    # Create the main output directory
    output_dir = os.path.join("results", f"chains_{sn_name}{identifier}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running {output_prefix} nested sampling for {output_dir}...")
    algo = blackjax.ns.adaptive.nss(
        logprior_fn=logprior_anomaly if ll_fn == loglikelihood_anomaly else logprior_standard,
        loglikelihood_fn=ll_fn,
        n_delete=NS_SETTINGS['n_delete'],
        num_mcmc_steps=num_mcmc_steps,
    )
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'], ll_fn)
    print("Initial particles generated, shape: ", initial_particles.shape)
    if ll_fn == loglikelihood_standard:
        state = algo.init(initial_particles, compute_batch_loglikelihood_standard)
    else:
        state = algo.init(initial_particles, compute_batch_loglikelihood_anomaly)
    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point
    dead = []
    emax_values = []  # For anomaly detection runs
    for i in tqdm.trange(num_iterations):
        if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
            break
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        if ll_fn == loglikelihood_anomaly:
            _, emax = compute_single_loglikelihood_anomaly(dead_info.particles[0])
            emax_values.append(emax)
        if i % 10 == 0:
            print(f"Iteration {i}: logZ = {state.sampler_state.logZ:.2f}")
    dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
    logw = log_weights(rng_key, dead)
    logZs = jax.scipy.special.logsumexp(logw, axis=0)
    print(f"Runtime evidence: {state.sampler_state.logZ:.2f}")
    print(f"Estimated evidence: {logZs.mean():.2f} +- {logZs.std():.2f}")
    if ll_fn == loglikelihood_standard:
        if fix_z:
            param_names = ['t0', 'log_x0', 'x1', 'c']
            if fit_sigma:
                param_names.append('sigma')
        else:
            param_names = ['z', 't0', 'log_x0', 'x1', 'c']
            if fit_sigma:
                param_names.append('sigma')
    else:
        if fix_z:
            param_names = ['t0', 'log_x0', 'x1', 'c']
            if fit_sigma:
                param_names.append('sigma')
            param_names.append('log_p')
        else:
            param_names = ['z', 't0', 'log_x0', 'x1', 'c']
            if fit_sigma:
                param_names.append('sigma')
            param_names.append('log_p')
    
    # Save chains with the correct filename
    chains_filename = f"{output_prefix}_dead-birth.txt"
    final_path = os.path.join(output_dir, chains_filename)
    
    # Extract data from dead info
    points = np.array(dead.particles)
    logls_death = np.array(dead.logL)
    logls_birth = np.array(dead.logL_birth)
    
    # Combine data: parameters, death likelihood, birth likelihood
    data = np.column_stack([points, logls_death, logls_birth])
    
    # Save directly to final location
    np.savetxt(final_path, data)
    print(f"Saved {data.shape[0]} samples to {final_path}")
    
    if ll_fn == loglikelihood_anomaly and emax_values:
        emax_array = jnp.stack(emax_values)
        print(f"emax_array shape: {emax_array.shape}")
        weights = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
        weights = weights[:, 0]
        weighted_emax = jnp.zeros(emax_array.shape[1])
        for i in range(emax_array.shape[1]):
            weighted_emax = weighted_emax.at[i].set(jnp.sum(emax_array[:, i] * weights) / jnp.sum(weights))
        print(f"weighted_emax shape: {weighted_emax.shape}")
        emax_output_path = os.path.join(output_dir, f"{output_prefix}_weighted_emax.txt")
        np.savetxt(emax_output_path, weighted_emax)
        print(f"Saved weighted emax values to {emax_output_path}")

def get_n_params(ll_fn):
    """Get the number of parameters being fit."""
    if ll_fn == loglikelihood_standard:
        if fix_z:
            return 5 if fit_sigma else 4
        else:
            return 6 if fit_sigma else 5
    else:
        if fix_z:
            return 6 if fit_sigma else 5
        else:
            return 7 if fit_sigma else 6

if __name__ == "__main__":
    # Add an identifier for this run (e.g. date, version, etc)
    identifier = "_tes"  # You can modify this or pass it as a command line argument
    
    print("\nRunning anomaly detection version...")
    n_params = get_n_params(loglikelihood_anomaly)
    num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']
    anomaly_samples = run_nested_sampling(loglikelihood_anomaly, "chains_anomaly", sn_name, identifier)

    print("Running standard version...")
    n_params = get_n_params(loglikelihood_standard)
    num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']
    standard_samples = run_nested_sampling(loglikelihood_standard, "chains_standard", sn_name, identifier)

    print("\nGenerating plots...")
    # Define output directory
    output_dir = f'results/chains_{sn_name}{identifier}'

    # Define parameter names based on settings
    if fix_z:
        base_params = ['t0', 'log_x0', 'x1', 'c']
    else:
        base_params = ['z', 't0', 'log_x0', 'x1', 'c']

    if fit_sigma:
        base_params.append('sigma')

    # Try to load weighted emax values and create initial plot
    try:
        weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
        
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(weighted_emax)), weighted_emax, 'k-', linewidth=2)
        plt.fill_between(np.arange(len(weighted_emax)), 0, weighted_emax, alpha=0.3)
        plt.xlabel('Data Point Number')
        plt.ylabel('Weighted Emax')
        plt.title('Weighted Emax by Data Point')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/weighted_emax.png', dpi=300, bbox_inches='tight')
        plt.close()
    except FileNotFoundError:
        print("Warning: Weighted emax file not found - skipping initial emax plot")

    # Try to load standard chains
    try:
        standard_samples = read_chains(f'{output_dir}/chains_standard', columns=base_params)
        have_standard = True
    except FileNotFoundError:
        print("Warning: Standard chains not found - some plots will be incomplete")
        have_standard = False
        standard_samples = None

    # Try to load anomaly chains with log_p parameter
    try:
        anomaly_params = base_params + ['log_p']
        anomaly_samples = read_chains(f'{output_dir}/chains_anomaly', columns=anomaly_params)
        have_anomaly = True
    except FileNotFoundError:
        print("Warning: Anomaly chains not found - some plots will be incomplete")
        have_anomaly = False
        anomaly_samples = None

    # Use the appropriate parameter names for plotting
    param_names = base_params  # Only plot the common parameters between both chains

    # Only create corner plot if we have at least one set of chains
    if have_standard or have_anomaly:
        # Create overlaid corner plot
        try:
            fig, axes = make_2d_axes(param_names, figsize=(10, 10), facecolor='w')
            
            # Plot standard samples if available
            if have_standard:
                try:
                    standard_samples.plot_2d(axes, alpha=0.7, label="Standard")
                except Exception as e:
                    print(f"Warning: Failed to plot standard samples in corner plot - {str(e)}")
            
            # Plot anomaly samples if available
            if have_anomaly:
                try:
                    anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
                except Exception as e:
                    print(f"Warning: Failed to plot anomaly samples in corner plot - {str(e)}")
            
            axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)
            plt.savefig(f'{output_dir}/corner_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to create corner comparison plot - {str(e)}")

    # Create anomaly corner plot only if we have anomaly chains
    if have_anomaly and 'log_p' in anomaly_samples.columns:
        try:
            log_p_params = base_params + ['log_p']
            fig, axes = make_2d_axes(log_p_params, figsize=(12, 12), facecolor='w')
            anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
            plt.suptitle('Anomaly Detection Corner Plot (including log_p)', fontsize=14)
            axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)
            plt.savefig(f'{output_dir}/corner_anomaly_logp.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to create anomaly corner plot - {str(e)}")

    def get_model_curve(samples, percentile=50):
        """Get model curve for given percentile of parameters."""
        params = {}
        for param in param_names:
            if param != 'log_p':  # Skip logp as it's not needed for the model
                params[param] = float(np.percentile(samples[param], percentile))
        if 'log_x0' in params:
            params['x0'] = 10**params['log_x0']
            del params['log_x0']  # Remove log_x0 as we now have x0
        if fix_z:
            params['z'] = fixed_z[0]
        if not fit_sigma:
            params['sigma'] = 1.0  # Add default sigma if not fitted
        return params

    try:
        # Create time grid for smooth model curves
        t_min = np.min(times) - 5
        t_max = np.max(times) + 5
        t_grid = np.linspace(t_min, t_max, 100)

        # Get unique bands
        unique_bands = np.unique(band_indices)
        n_bands = len(unique_bands)

        # Set up the plot with two subplots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.1)

        # Main light curve plot
        ax1 = plt.subplot(gs[0])

        # Define colours for each band
        default_colours = ['g', 'c', 'orange', 'r']
        default_markers = ['o', 's', 'D', '^']
        
        # Ensure we have enough colours and markers for all bands
        if n_bands > len(default_colours):
            colours = plt.cm.tab10(np.linspace(0, 1, n_bands))
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '8'][:n_bands]
        else:
            colours = default_colours[:n_bands]
            markers = default_markers[:n_bands]

        # Load weighted emax values first to identify anomalous points
        try:
            weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
            plotting_threshold = 0.2
            
            # Count total anomalous points
            total_anomalous_points = 0
            
            # Create time points that match the actual data points
            all_times = np.sort(np.unique(times))
            if len(all_times) != len(weighted_emax):
                print(f"Warning: Number of unique time points ({len(all_times)}) "
                      f"doesn't match number of weighted_emax values ({len(weighted_emax)})")
            
            # Create time points for the emax plot - using actual data time points
            emax_times = all_times
        except FileNotFoundError:
            print("Warning: Weighted emax file not found")
            weighted_emax = None

        # Plot data points for each band
        try:
            for i, band_idx in enumerate(unique_bands):
                mask = band_indices == band_idx
                band_times = times[mask]
                band_fluxes = fluxes[mask]
                band_errors = fluxerrs[mask]

                if weighted_emax is not None:
                    # Map each data point to its index in the sorted unique times
                    time_indices = np.searchsorted(all_times, band_times)
                    # Ensure indices are within bounds
                    time_indices = np.clip(time_indices, 0, len(weighted_emax) - 1)
                    # Get the emax value for each point
                    point_emax = weighted_emax[time_indices]
                    
                    # Determine which points are anomalous
                    normal_mask = point_emax >= plotting_threshold
                    anomaly_mask = point_emax < plotting_threshold
                    
                    # Update total count
                    total_anomalous_points += np.sum(anomaly_mask)

                    # Plot normal points
                    if np.any(normal_mask):
                        ax1.errorbar(band_times[normal_mask], band_fluxes[normal_mask], 
                                   yerr=band_errors[normal_mask],
                                   fmt=markers[i], color=colours[i], 
                                   label=f'Band {i} Data',
                                   markersize=8, alpha=0.6)
                    
                    # Plot anomalous points with star markers
                    if np.any(anomaly_mask):
                        label = f'Band {i} Anomalous' if np.any(normal_mask) else f'Band {i} Data'
                        ax1.errorbar(band_times[anomaly_mask], band_fluxes[anomaly_mask], 
                                   yerr=band_errors[anomaly_mask],
                                   fmt='*', color=colours[i], 
                                   label=label,
                                   markersize=15, alpha=0.8)
                else:
                    # Plot all points normally if no weighted_emax available
                    ax1.errorbar(band_times, band_fluxes, yerr=band_errors,
                               fmt=markers[i], color=colours[i], label=f'Band {i} Data',
                               markersize=8, alpha=0.6)
        except Exception as e:
            print(f"Warning: Failed to plot some data points - {str(e)}")

        # Calculate and plot model curves for both standard and anomaly if available
        if have_standard or have_anomaly:
            for name, samples, has_samples in [
                ("Standard", standard_samples, have_standard), 
                ("Anomaly", anomaly_samples, have_anomaly)
            ]:
                if has_samples:
                    try:
                        params = get_model_curve(samples)
                        linestyle = '--' if name == "Standard" else '-'
                        
                        for i in range(n_bands):
                            try:
                                # Calculate model fluxes
                                model_fluxes = optimized_salt3_multiband_flux(
                                    jnp.array(t_grid),
                                    bridges,
                                    params,
                                    zps=zps,
                                    zpsys='ab'
                                )
                                
                                # Extract fluxes for this band
                                band_fluxes = model_fluxes[:, i]
                                
                                # Plot model curve
                                ax1.plot(t_grid, band_fluxes, linestyle, color=colours[i], 
                                        label=f'Band {i} {name}', linewidth=2, alpha=0.8)
                            except Exception as e:
                                print(f"Warning: Failed to plot {name} model curve for band {i} - {str(e)}")
                                continue
                    except Exception as e:
                        print(f"Warning: Failed to plot {name} model curves - {str(e)}")
                        continue

        # Add labels and title to main plot
        ax1.set_xlabel('MJD', fontsize=12)
        ax1.set_ylabel('Flux', fontsize=12)
        title = 'Light Curve Fit Comparison'
        if fix_z:
            title += f' (z = {fixed_z[0]:.4f})'
        ax1.set_title(title, fontsize=14)

        # Add legend to main plot
        ax1.legend(ncol=2, fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Try to load weighted emax values and create subplot
        try:
            weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
            ax2 = plt.subplot(gs[1])
            
            # Use actual data time points for the emax plot
            ax2.plot(all_times, weighted_emax, 'k-', linewidth=2)
            ax2.fill_between(all_times, 0, weighted_emax, alpha=0.3, color='gray')
            ax2.set_xlabel('MJD', fontsize=12)
            ax2.set_ylabel('Emax', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Add horizontal line at threshold
            ax2.axhline(y=plotting_threshold, color='r', linestyle='--', alpha=0.5, 
                       label=f'Plotting threshold ({plotting_threshold}) - {total_anomalous_points} points below')
            ax2.legend()

            # Ensure x-axis limits match between plots
            xlim = ax1.get_xlim()
            ax2.set_xlim(xlim)

            # Remove x-axis labels from top plot
            ax1.set_xlabel('')
        except FileNotFoundError:
            print("Warning: Weighted emax file not found - skipping emax subplot")
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Failed to create emax subplot - {str(e)}")
            plt.tight_layout()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{output_dir}/light_curve_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Warning: Failed to create light curve comparison plot - {str(e)}")
        plt.close('all')

    # Save parameter statistics to a text file if chains are available
    if have_standard or have_anomaly:
        stats_text = ["Parameter Statistics Comparison:", "-" * 50]
        for param in param_names:
            if have_standard:
                std_mean = standard_samples[param].mean()
                std_std = standard_samples[param].std()
                stats_text.append(f"\n{param}:")
                stats_text.append(f"  Standard: {std_mean:.6f} ± {std_std:.6f}")
            if have_anomaly:
                anom_mean = anomaly_samples[param].mean()
                anom_std = anomaly_samples[param].std()
                stats_text.append(f"  Anomaly:  {anom_mean:.6f} ± {anom_std:.6f}")

        # Add log_p statistics for anomaly case if available
        if have_anomaly and 'log_p' in anomaly_samples.columns:
            stats_text.extend([
                "\nlog_p (Anomaly only):",
                f"  Mean: {anomaly_samples['log_p'].mean():.6f} ± {anomaly_samples['log_p'].std():.6f}",
                f"  Max: {anomaly_samples['log_p'].max():.6f}",
                f"  Min: {anomaly_samples['log_p'].min():.6f}"
            ])

        # Save statistics
        stats_text = '\n'.join(stats_text)
        with open(f'{output_dir}/parameter_statistics.txt', 'w') as f:
            f.write(stats_text)
    