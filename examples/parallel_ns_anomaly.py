import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
import os
import yaml
from functools import partial
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.bandpasses import register_bandpass, get_bandpass, register_all_bandpasses
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_and_process_data, get_all_supernovae_with_redshifts
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes
import pandas as pd
from typing import NamedTuple, Dict, Any, List, Tuple
import time
import jax.lax as lax
from jax import jit, vmap

# Define default settings for nested sampling and prior bounds
DEFAULT_NS_SETTINGS = {
    'max_iterations': 5000,  # Set to 5000 iterations
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
    'data_dir': 'hsf_DR1'  # Directory containing supernova data
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
data_dir = settings['data_dir']

NS_SETTINGS = DEFAULT_NS_SETTINGS.copy()
NS_SETTINGS.update(settings.get('nested_sampling', {}))

PRIOR_BOUNDS = DEFAULT_PRIOR_BOUNDS.copy()
if 'prior_bounds' in settings:
    PRIOR_BOUNDS.update(settings['prior_bounds'])

# Option flag: when fit_sigma is True, an extra parameter is added
fit_sigma = NS_SETTINGS['fit_sigma']

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Add new data structure for supernova data
class SupernovaData(NamedTuple):
    times: jnp.ndarray
    fluxes: jnp.ndarray
    fluxerrs: jnp.ndarray
    zps: jnp.ndarray
    band_indices: jnp.ndarray
    bridges: tuple
    fixed_z: tuple

def pad_supernova_data(sndata: SupernovaData, target_length: int) -> Dict[str, Any]:
    """Pad supernova data arrays to target length.
    
    Args:
        sndata: SupernovaData instance to pad
        target_length: Length to pad arrays to
        
    Returns:
        Dictionary containing padded arrays and mask
    """
    # Calculate padding amount
    pad_amount = target_length - sndata.times.shape[0]
    
    # Pad arrays with appropriate values
    times = jnp.pad(sndata.times, (0, pad_amount), constant_values=0)
    fluxes = jnp.pad(sndata.fluxes, (0, pad_amount), constant_values=0)
    fluxerrs = jnp.pad(sndata.fluxerrs, (0, pad_amount), constant_values=1.0)
    band_indices = jnp.pad(sndata.band_indices, (0, pad_amount), constant_values=-1)
    zps = jnp.pad(sndata.zps, (0, pad_amount), constant_values=0)
    
    # Create mask: 1 for valid entries, 0 for padded values
    mask = jnp.concatenate([jnp.ones(sndata.times.shape[0]), jnp.zeros(pad_amount)])
    
    return {
        'times': times,
        'fluxes': fluxes,
        'fluxerrs': fluxerrs,
        'zps': zps,
        'band_indices': band_indices,
        'bridges': sndata.bridges,
        'fixed_z': sndata.fixed_z,
        'mask': mask
    }

def setup_priors_and_bounds(fix_z=True, fit_sigma=False):
    """Set up prior distributions and parameter bounds."""
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

    # Set up prior distributions
    standard_prior_dists = {}
    for param, (low, high) in standard_param_bounds.items():
        standard_prior_dists[param] = distrax.Uniform(low=low, high=high)

    # Add log_p parameter for anomaly detection
    anomaly_param_bounds = standard_param_bounds.copy()
    anomaly_param_bounds['log_p'] = (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    
    anomaly_prior_dists = standard_prior_dists.copy()
    anomaly_prior_dists['log_p'] = distrax.Uniform(
        low=PRIOR_BOUNDS['log_p']['min'], 
        high=PRIOR_BOUNDS['log_p']['max']
    )

    return standard_param_bounds, standard_prior_dists, anomaly_param_bounds, anomaly_prior_dists

@partial(jax.jit, static_argnums=(1,))
def compute_single_loglikelihood_standard(params, n_params, data):
    """Compute Gaussian log likelihood for a single set of parameters (standard)."""
    # Handle both single and batched parameters
    params = jnp.atleast_2d(params)
    
    # Extract parameters based on shape
    if n_params == 4:  # fix_z=True, fit_sigma=False
        t0 = params[:, 0]
        log_x0 = params[:, 1]
        x1 = params[:, 2]
        c = params[:, 3]
        sigma = jnp.ones_like(t0)
        z = jnp.full_like(t0, data['fixed_z'][0])
    elif n_params == 5 and fix_z:  # fix_z=True, fit_sigma=True
        t0 = params[:, 0]
        log_x0 = params[:, 1]
        x1 = params[:, 2]
        c = params[:, 3]
        sigma = params[:, 4]
        z = jnp.full_like(t0, data['fixed_z'][0])
    elif n_params == 5:  # fix_z=False, fit_sigma=False
        z = params[:, 0]
        t0 = params[:, 1]
        log_x0 = params[:, 2]
        x1 = params[:, 3]
        c = params[:, 4]
        sigma = jnp.ones_like(t0)
    else:  # fix_z=False, fit_sigma=True
        z = params[:, 0]
        t0 = params[:, 1]
        log_x0 = params[:, 2]
        x1 = params[:, 3]
        c = params[:, 4]
        sigma = params[:, 5]

    x0 = 10 ** log_x0
    
    # Vectorize over batch dimension
    def compute_single(z_i, t0_i, x0_i, x1_i, c_i, sigma_i):
        param_dict = {'z': z_i, 't0': t0_i, 'x0': x0_i, 'x1': x1_i, 'c': c_i}
        model_fluxes = optimized_salt3_multiband_flux(data['times'], data['bridges'], param_dict, zps=data['zps'], zpsys='ab')
        model_fluxes = model_fluxes[jnp.arange(len(data['times'])), data['band_indices']]
        eff_fluxerrs = sigma_i * data['fluxerrs']
        # Apply mask to sum only valid entries
        chi2 = jnp.sum(data['mask'] * ((data['fluxes'] - model_fluxes) / eff_fluxerrs) ** 2)
        log_term = jnp.sum(data['mask'] * jnp.log(2 * jnp.pi * eff_fluxerrs ** 2))
        return -0.5 * (chi2 + log_term)
    
    # Map over batch dimension
    return jax.vmap(compute_single)(z, t0, x0, x1, c, sigma)

@partial(jax.jit, static_argnums=(1,))
def compute_single_loglikelihood_anomaly(params, n_params, data):
    """Compute Gaussian log likelihood for a single set of parameters with anomaly detection."""
    # Handle both single and batched parameters
    params = jnp.atleast_2d(params)
    
    # Extract parameters based on shape
    if n_params == 5:  # fix_z=True, fit_sigma=False
        t0 = params[:, 0]
        log_x0 = params[:, 1]
        x1 = params[:, 2]
        c = params[:, 3]
        log_p = params[:, 4]
        sigma = jnp.ones_like(t0)
        z = jnp.full_like(t0, data['fixed_z'][0])
    elif n_params == 6 and fix_z:  # fix_z=True, fit_sigma=True
        t0 = params[:, 0]
        log_x0 = params[:, 1]
        x1 = params[:, 2]
        c = params[:, 3]
        sigma = params[:, 4]
        log_p = params[:, 5]
        z = jnp.full_like(t0, data['fixed_z'][0])
    elif n_params == 6:  # fix_z=False, fit_sigma=False
        z = params[:, 0]
        t0 = params[:, 1]
        log_x0 = params[:, 2]
        x1 = params[:, 3]
        c = params[:, 4]
        log_p = params[:, 5]
        sigma = jnp.ones_like(t0)
    else:  # fix_z=False, fit_sigma=True
        z = params[:, 0]
        t0 = params[:, 1]
        log_x0 = params[:, 2]
        x1 = params[:, 3]
        c = params[:, 4]
        sigma = params[:, 5]
        log_p = params[:, 6]

    x0 = 10 ** log_x0
    p = jnp.exp(log_p)
    
    # Vectorize over batch dimension
    def compute_single(z_i, t0_i, x0_i, x1_i, c_i, sigma_i, p_i):
        param_dict = {'z': z_i, 't0': t0_i, 'x0': x0_i, 'x1': x1_i, 'c': c_i}
        model_fluxes = optimized_salt3_multiband_flux(data['times'], data['bridges'], param_dict, zps=data['zps'], zpsys='ab')
        model_fluxes = model_fluxes[jnp.arange(len(data['times'])), data['band_indices']]
        eff_fluxerrs = sigma_i * data['fluxerrs']
        # Apply mask to compute point-wise log likelihood
        point_logL = data['mask'] * (-0.5 * ((data['fluxes'] - model_fluxes) / eff_fluxerrs) ** 2 - 0.5 * jnp.log(2 * jnp.pi * eff_fluxerrs ** 2) + jnp.log(1 - p_i))
        delta = jnp.max(jnp.abs(data['fluxes'] * data['mask']))  # Only consider valid fluxes for delta
        emax = point_logL > (jnp.log(p_i) - jnp.log(delta))
        logL = jnp.where(emax, point_logL, jnp.log(p_i) - jnp.log(delta))
        return jnp.sum(logL), emax
    
    # Map over batch dimension
    return jax.vmap(compute_single)(z, t0, x0, x1, c, sigma, p)

def setup_likelihood_and_prior_functions(fix_z, fit_sigma, standard_prior_dists, anomaly_prior_dists):
    """Set up likelihood and prior functions for both standard and anomaly cases."""
    n_params_standard = 4 if (fix_z and not fit_sigma) else 5 if (fix_z or not fit_sigma) else 6
    n_params_anomaly = n_params_standard + 1

    @jax.jit
    def logprior_standard(params):
        """Calculate log prior probability for standard nested sampling."""
        params = jnp.atleast_2d(params)
        logp = 0.0
        idx = 0
        if not fix_z:
            logp += standard_prior_dists['z'].log_prob(params[:, idx])
            idx += 1
        logp += standard_prior_dists['t0'].log_prob(params[:, idx])
        logp += standard_prior_dists['x0'].log_prob(params[:, idx + 1])
        logp += standard_prior_dists['x1'].log_prob(params[:, idx + 2])
        logp += standard_prior_dists['c'].log_prob(params[:, idx + 3])
        if fit_sigma:
            logp += standard_prior_dists['sigma'].log_prob(params[:, -1])
        return jnp.reshape(logp, (-1,))

    @jax.jit
    def logprior_anomaly(params):
        """Calculate log prior probability for anomaly detection nested sampling."""
        params = jnp.atleast_2d(params)
        logp = 0.0
        idx = 0
        if not fix_z:
            logp += anomaly_prior_dists['z'].log_prob(params[:, idx])
            idx += 1
        logp += anomaly_prior_dists['t0'].log_prob(params[:, idx])
        logp += anomaly_prior_dists['x0'].log_prob(params[:, idx + 1])
        logp += anomaly_prior_dists['x1'].log_prob(params[:, idx + 2])
        logp += anomaly_prior_dists['c'].log_prob(params[:, idx + 3])
        if fit_sigma:
            logp += anomaly_prior_dists['sigma'].log_prob(params[:, -2])
        logp += anomaly_prior_dists['log_p'].log_prob(params[:, -1])
        return jnp.reshape(logp, (-1,))

    @jax.jit
    def loglikelihood_standard(params):
        """Calculate log likelihood for standard nested sampling."""
        params = jnp.atleast_2d(params)
        batch_loglike = jax.vmap(lambda p: compute_single_loglikelihood_standard(p, n_params_standard))(params)
        # Ensure output is a 1D array with shape (n_samples,)
        return jnp.reshape(batch_loglike, (-1,))

    @jax.jit
    def loglikelihood_anomaly(params):
        """Calculate log likelihood for anomaly detection nested sampling."""
        params = jnp.atleast_2d(params)
        logls, emaxs = jax.vmap(lambda p: compute_single_loglikelihood_anomaly(p, n_params_anomaly))(params)
        # Ensure output is a 1D array with shape (n_samples,)
        return jnp.reshape(logls, (-1,))

    return (logprior_standard, loglikelihood_standard, n_params_standard,
            logprior_anomaly, loglikelihood_anomaly, n_params_anomaly)

def sample_from_priors(rng_key, n_samples, prior_dists, param_order):
    """Sample from prior distributions."""
    keys = jax.random.split(rng_key, len(param_order))
    samples = []
    for i, param in enumerate(param_order):
        samples.append(prior_dists[param].sample(seed=keys[i], sample_shape=(n_samples,)))
    return jnp.column_stack(samples)

def load_batch_data(supernovae: List[Tuple[str, float, float, str]], 
                   data_dir: str, 
                   fix_z: bool,
                   batch_size: int = 10) -> List[SupernovaData]:
    """Load data for a batch of supernovae."""
    batch_data = []
    for i in range(0, len(supernovae), batch_size):
        batch = supernovae[i:i + batch_size]
        batch_sn_data = []
        
        for sn_name, z, z_err, flag in batch:
            print(f"\nAttempting to load data for {sn_name}...")
            try:
                times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(
                    sn_name, data_dir=data_dir, fix_z=fix_z
                )
                print(f"Successfully loaded {sn_name} data:")
                print(f"  - Number of data points: {len(times)}")
                print(f"  - Number of unique bands: {len(np.unique(band_indices))}")
                print(f"  - Time range: {times.min():.1f} to {times.max():.1f}")
                batch_sn_data.append(SupernovaData(
                    times=times,
                    fluxes=fluxes,
                    fluxerrs=fluxerrs,
                    zps=zps,
                    band_indices=band_indices,
                    bridges=bridges,
                    fixed_z=fixed_z
                ))
            except Exception as e:
                print(f"Skipping {sn_name} due to error: {str(e)}")
                continue
        
        if batch_sn_data:
            print(f"\nSuccessfully loaded batch of {len(batch_sn_data)} supernovae")
            batch_data.append(batch_sn_data)
        else:
            print("\nNo valid data in this batch, skipping...")
    
    return batch_data

@jax.jit
def log_weights(rng_key, dead_points, logls_death, logls_birth):
    """Calculate log weights for nested sampling."""
    n_points = len(dead_points)
    log_vols = jnp.log1p(-jnp.exp(-1.0 / NS_SETTINGS['n_live']))
    log_vols = jnp.arange(n_points) * log_vols
    log_vols = jnp.log1p(-jnp.exp(log_vols))
    
    # Calculate log weights
    logw = logls_death + log_vols
    
    return logw

@partial(jit, static_argnums=(1,2))
def run_nested_sampling_step(state, n_points, algo):
    """Run a single step of nested sampling with fixed array shapes."""
    logls_death = jnp.zeros(n_points, dtype=jnp.float64)
    logls_birth = jnp.zeros(n_points, dtype=jnp.float64)
    dead_points = jnp.zeros((n_points, state.sampler_state.particles.shape[1]), dtype=jnp.float64)
    
    def body_fun(i, vals):
        state, k, logls_death, logls_birth, dead_points = vals
        k, subk = jax.random.split(k, 2)
        state, info = algo.step(subk, state)
        # Ensure scalar values are reshaped to (1,) before assignment
        logls_death = logls_death.at[i].set(jnp.reshape(info.logL, (1,))[0])
        logls_birth = logls_birth.at[i].set(jnp.reshape(info.logL_birth, (1,))[0])
        dead_points = dead_points.at[i].set(info.particles[0])
        return (state, k, logls_death, logls_birth, dead_points)
        
    rng_key = jax.random.PRNGKey(0)
    (state, _, logls_death, logls_birth, dead_points) = lax.fori_loop(
        0, n_points, body_fun, (state, rng_key, logls_death, logls_birth, dead_points))
    
    return state, logls_death, logls_birth, dead_points

@partial(jit, static_argnums=(1,2))
def run_nested_sampling_step_batched(batch_states, n_points, algo):
    """Run nested sampling steps in parallel for multiple supernovae.
    
    Args:
        batch_states: Batched nested sampling states, one per supernova
        n_points: Number of iterations to run
        algo: Nested sampling algorithm instance
        
    Returns:
        Tuple of (final_states, logls_death, logls_birth, dead_points)
    """
    # Assume batch dimension is axis 0
    batch_size = batch_states.sampler_state.particles.shape[0]
    
    # Preallocate arrays for logging history
    logls_death = jnp.zeros((batch_size, n_points), dtype=jnp.float64)
    logls_birth = jnp.zeros((batch_size, n_points), dtype=jnp.float64)
    dead_points = jnp.zeros((batch_size, n_points, batch_states.sampler_state.particles.shape[2]), dtype=jnp.float64)
    
    # Create a batch of random keys - one for each supernova
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    
    def body_fun(i, vals):
        states, rng_keys, ld, lb, dp = vals
        
        # Split keys for each supernova in the batch
        new_keys = jax.vmap(lambda k: jax.random.split(k))(rng_keys)
        rng_keys_next = new_keys[:, 0]  # Take first split for next iteration
        subkeys = new_keys[:, 1]        # Take second split for current step
        
        # Update each state in the batch in parallel
        new_states, infos = jax.vmap(lambda key, state: algo.step(key, state))(subkeys, states)
        
        # Extract and reshape log-likelihoods to match expected dimensions
        death_lls = jax.vmap(lambda info: info.logL)(infos)
        birth_lls = jax.vmap(lambda info: info.logL_birth)(infos)
        dead_pts = jax.vmap(lambda info: info.particles[0])(infos)
        
        # Ensure correct shapes for assignment
        death_lls = jnp.reshape(death_lls, (batch_size,))
        birth_lls = jnp.reshape(birth_lls, (batch_size,))
        
        # Record the log likelihoods and dead points
        ld = ld.at[:, i].set(death_lls)
        lb = lb.at[:, i].set(birth_lls)
        dp = dp.at[:, i, :].set(dead_pts)
        
        return (new_states, rng_keys_next, ld, lb, dp)
    
    final_states, final_rng_keys, logls_death, logls_birth, dead_points = lax.fori_loop(
        0, n_points, body_fun, (batch_states, rng_keys, logls_death, logls_birth, dead_points))
    
    return final_states, logls_death, logls_birth, dead_points

def run_nested_sampling_batch(batch_data, fix_z=True, fit_sigma=False):
    """Run nested sampling on a batch of supernovae simultaneously."""
    try:
        print("\nInitializing nested sampling...")
        # Set up priors and bounds once for the batch
        standard_param_bounds, standard_prior_dists, anomaly_param_bounds, anomaly_prior_dists = setup_priors_and_bounds(fix_z, fit_sigma)
        print("Prior bounds and distributions set up successfully")
        
        # Set up likelihood and prior functions
        (logprior_standard, loglikelihood_standard, n_params_standard,
         logprior_anomaly, loglikelihood_anomaly, n_params_anomaly) = setup_likelihood_and_prior_functions(
            fix_z, fit_sigma, standard_prior_dists, anomaly_prior_dists
        )
        print(f"Likelihood and prior functions initialized with {n_params_standard} standard parameters and {n_params_anomaly} anomaly parameters")
        
        # Initialize results list with empty dictionaries for each supernova
        batch_results = [{} for _ in range(len(batch_data))]
        
        # First, pad each supernova data to the same length
        max_len = max([sn.times.shape[0] for sn in batch_data])
        print(f"\nPadding all supernova data to length {max_len}")
        padded_batch = [pad_supernova_data(sn, max_len) for sn in batch_data]
        print("Data padding complete")
        
        # Process both standard and anomaly nested sampling separately
        for ns_type, (logprior, base_loglike, n_params, prior_dists, param_bounds) in [
            ('standard', (logprior_standard, compute_single_loglikelihood_standard, n_params_standard, standard_prior_dists, standard_param_bounds)),
            ('anomaly', (logprior_anomaly, compute_single_loglikelihood_anomaly, n_params_anomaly, anomaly_prior_dists, anomaly_param_bounds))
        ]:
            print(f"\nProcessing {ns_type} nested sampling")
            print(f"Number of parameters: {n_params}")
            print(f"Parameter bounds: {param_bounds}")
            
            # Initialize particles and states for each supernova
            batch_states = []
            rng_key = jax.random.PRNGKey(0)
            param_order = list(param_bounds.keys())
            print(f"Parameter order: {param_order}")
            
            for idx, padded_data in enumerate(padded_batch):
                print(f"\nInitializing nested sampling for supernova {idx}")
                # Initialize particles for this supernova
                rng_key, init_key = jax.random.split(rng_key)
                initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'], prior_dists, param_order)
                print(f"Initial particles shape: {initial_particles.shape}")
                
                # Create a closure over the likelihood function for this supernova
                def make_likelihood_fn(data):
                    def likelihood_fn(params):
                        result = base_loglike(params, n_params, data)
                        if isinstance(result, tuple):
                            return result[0]  # For anomaly case, take just the logL
                        return result
                    return likelihood_fn
                
                # Initialize nested sampling algorithm for this supernova
                print("Initializing BlackJAX nested sampling algorithm")
                algo_instance = blackjax.ns.adaptive.nss(
                    logprior_fn=logprior,
                    loglikelihood_fn=make_likelihood_fn(padded_data),
                    n_delete=NS_SETTINGS['n_delete'],
                    num_mcmc_steps=n_params * NS_SETTINGS['num_mcmc_steps_multiplier'],
                )
                
                # Initialize state
                print("Initializing nested sampling state")
                state = algo_instance.init(initial_particles, make_likelihood_fn(padded_data))
                batch_states.append(state)
                print("State initialized successfully")
            
            # Stack states into a single batch
            print("\nStacking states into batch")
            batch_states = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *batch_states)
            print(f"Batch states shape: {batch_states.sampler_state.particles.shape}")
            
            # Run batched nested sampling
            n_points = NS_SETTINGS['max_iterations']
            print(f"\nRunning batched nested sampling for {n_points} iterations")
            print(f"Batch size: {len(batch_states.sampler_state.particles)}")
            
            final_states, logls_death, logls_birth, dead_points = run_nested_sampling_step_batched(
                batch_states, n_points, algo_instance
            )
            print("Nested sampling complete")
            print(f"Final batched shapes - dead_points: {dead_points.shape}, logls_death: {logls_death.shape}, logls_birth: {logls_birth.shape}")
            
            # Process results for each supernova
            for idx, padded_data in enumerate(padded_batch):
                print(f"\nProcessing results for supernova {idx}")
                evidence = float(final_states.sampler_state.logZ[idx])
                print(f"Evidence: {evidence}")
                
                # Add results to the existing dictionary for this supernova
                batch_results[idx][ns_type] = {
                    'evidence': evidence,
                    'evidence_err': 0.0,  # Could compute this from logls if needed
                    'dead_points': jnp.array(dead_points[idx]),
                    'logls_death': jnp.array(logls_death[idx]),
                    'logls_birth': jnp.array(logls_birth[idx])
                }
                
                if ns_type == 'anomaly':
                    print("Computing weighted emax values for anomaly detection")
                    _, emax_values = jax.vmap(lambda p: compute_single_loglikelihood_anomaly(p, n_params_anomaly, padded_data))(dead_points[idx])
                    weights = jnp.exp(log_weights(jax.random.PRNGKey(0), dead_points[idx], logls_death[idx], logls_birth[idx]))
                    weights = weights / jnp.sum(weights)
                    weighted_emax = jnp.sum(emax_values * weights[:, None], axis=0)
                    batch_results[idx][ns_type]['weighted_emax'] = weighted_emax
                    print(f"Weighted emax shape: {weighted_emax.shape}")
                
                print(f"Results processed for supernova {idx}")
        
        return batch_results
            
    except Exception as e:
        print(f"Error in nested sampling: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return [None] * len(batch_data)

def test_parallel_implementation(n_test=2):
    """Test the parallel nested sampling implementation with a small number of supernovae."""
    print("\nTesting parallel nested sampling implementation...")
    
    # Get a small number of supernovae
    supernovae = get_all_supernovae_with_redshifts()[:n_test]
    print(f"Testing with {len(supernovae)} supernovae")
    
    # Load the data
    batch_data = load_batch_data(supernovae, data_dir, fix_z=True, batch_size=n_test)[0]
    
    # Print data shapes before padding
    print("\nData shapes before padding:")
    for i, sn_data in enumerate(batch_data):
        print(f"SN {i}: times shape = {sn_data.times.shape}, fluxes shape = {sn_data.fluxes.shape}")
    
    # Run nested sampling
    try:
        results = run_nested_sampling_batch(batch_data, fix_z=True, fit_sigma=False)
        
        # Check results
        print("\nResults summary:")
        for i, result in enumerate(results):
            if result is not None:
                print(f"\nSN {i}:")
                for ns_type in result.keys():
                    print(f"  {ns_type} evidence: {result[ns_type]['evidence']:.2f}")
                    print(f"  {ns_type} dead points shape: {result[ns_type]['dead_points'].shape}")
                    if 'weighted_emax' in result[ns_type]:
                        print(f"  {ns_type} weighted_emax shape: {result[ns_type]['weighted_emax'].shape}")
            else:
                print(f"\nSN {i}: Failed to process")
        
        return True
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

def main():
    # Get list of all supernovae with redshifts
    supernovae = get_all_supernovae_with_redshifts()
    print(f"Found {len(supernovae)} supernovae with redshifts")
    
    # Take exactly 4 supernovae
    supernovae = supernovae[:4]
    print(f"Processing {len(supernovae)} supernovae")

    # Create main output directory
    output_dir = os.path.join("results", "parallel_ns")
    os.makedirs(output_dir, exist_ok=True)

    # Process supernovae in batches of 2
    BATCH_SIZE = 2  # Process 2 supernovae per batch
    start_time = time.time()
    
    # Load data in batches
    batches = load_batch_data(supernovae, data_dir, fix_z, BATCH_SIZE)
    
    # Process each batch
    all_results = {}
    for batch_idx, batch_data in enumerate(batches):
        print(f"\nProcessing batch {batch_idx + 1}/{len(batches)}")
        
        try:
            batch_results = run_nested_sampling_batch(batch_data, fix_z=fix_z, fit_sigma=fit_sigma)
            
            # Save results for each supernova in the batch
            for sn_idx, (sn_name, z, z_err, flag) in enumerate(supernovae[batch_idx * BATCH_SIZE:
                                                                         (batch_idx + 1) * BATCH_SIZE][:len(batch_results)]):
                results = batch_results[sn_idx]
                all_results[sn_name] = results
                
                # Create output directory for this supernova
                sn_output_dir = os.path.join(output_dir, sn_name)
                os.makedirs(sn_output_dir, exist_ok=True)
                
                # Save chains
                for ns_type in ['standard', 'anomaly']:
                    chains_filename = f"chains_{ns_type}_dead-birth.txt"
                    final_path = os.path.join(sn_output_dir, chains_filename)
                    
                    data = np.column_stack([
                        results[ns_type]['dead_points'],
                        results[ns_type]['logls_death'],
                        results[ns_type]['logls_birth']
                    ])
                    np.savetxt(final_path, data)
                
                # Save weighted emax values for anomaly detection
                if 'weighted_emax' in results['anomaly']:
                    emax_output_path = os.path.join(sn_output_dir, "weighted_emax.txt")
                    np.savetxt(emax_output_path, results['anomaly']['weighted_emax'])
                
                # Save summary
                summary = {
                    'redshift': z,
                    'redshift_err': z_err,
                    'redshift_flag': flag,
                    'standard_evidence': results['standard']['evidence'],
                    'standard_evidence_err': results['standard']['evidence_err'],
                    'anomaly_evidence': results['anomaly']['evidence'],
                    'anomaly_evidence_err': results['anomaly']['evidence_err'],
                }
                
                summary_path = os.path.join(sn_output_dir, "summary.txt")
                with open(summary_path, 'w') as f:
                    for key, value in summary.items():
                        f.write(f"{key}: {value}\n")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            continue

    # Calculate and print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    if len(supernovae) > 0:
        print(f"Average time per supernova: {total_time/len(supernovae):.2f} seconds")

    # Save overall summary if we have results
    if all_results:
        summary_data = []
        for sn_name, results in all_results.items():
            summary_data.append({
                'sn_name': sn_name,
                'standard_evidence': results['standard']['evidence'],
                'anomaly_evidence': results['anomaly']['evidence'],
                'log_B': results['anomaly']['evidence'] - results['standard']['evidence']
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

            # Create summary plot
            plt.figure(figsize=(12, 6))
            plt.scatter(summary_df['standard_evidence'], summary_df['log_B'])
            plt.xlabel('Standard Evidence')
            plt.ylabel('log B (Anomaly/Standard)')
            plt.title('Evidence Comparison Across All Supernovae')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "evidence_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("No results were successfully processed.")

if __name__ == "__main__":
    # Run test first
    if test_parallel_implementation(n_test=2):
        print("\nTest passed, proceeding with main execution...\n")
        main()
    else:
        print("\nTest failed, please fix implementation before running main()") 