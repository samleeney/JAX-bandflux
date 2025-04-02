import distrax
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
import yaml
import pandas as pd
from blackjax.ns.utils import log_weights
from functools import partial # Added import
from jax_supernovae.salt3 import optimized_salt3_multiband_flux # Removed precompute_bandflux_bridge import as it's not directly used here
from jax_supernovae.bandpasses import register_bandpass, get_bandpass, register_all_bandpasses, Bandpass
from jax_supernovae.utils import save_chains_dead_birth
# Import the updated load_and_process_data
from jax_supernovae.data import load_redshift, load_hsf_data, load_and_process_data
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes
import requests
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".2"


# Define default settings for nested sampling and prior bounds
DEFAULT_NS_SETTINGS = {
    'max_iterations': int(os.environ.get('NS_MAX_ITERATIONS', '500')), # Default 500 iterations
    'n_delete': 75,
    'n_live': 150,
    'num_mcmc_steps_multiplier': 5,
    'fit_sigma': False,
    'fit_log_p': True, # Default to fitting anomaly model
    'fit_z': True # Default to fixing redshift
}

DEFAULT_PRIOR_BOUNDS = {
    'z': {'min': 0.001, 'max': 0.2},
    't0': {'min': 58000.0, 'max': 60000.0},
    'x0': {'min': -5.0, 'max': -1}, # log10(x0) bounds
    'x1': {'min': -10, 'max': 10},
    'c': {'min': -0.6, 'max': 0.6},
    'sigma': {'min': 0.001, 'max': 5}, # sigma bounds (not log_sigma)
    'log_p': {'min': -20, 'max': -1} # log10(p) bounds
}

# Default settings
DEFAULT_SETTINGS = {
    'fix_z': True,
    'sn_name': '19vnk',  # Default supernova to analyze
    'selected_bandpasses': None,  # Default: use all available bandpasses
    'custom_bandpass_files': None  # Default: no custom bandpass files
}

# Try to load settings.yaml; if not found, use an empty dictionary
try:
    with open('settings.yaml', 'r') as f:
        settings_from_file = yaml.safe_load(f)
except FileNotFoundError:
    settings_from_file = {}

# Merge the settings from file with the defaults
settings = DEFAULT_SETTINGS.copy()
settings.update(settings_from_file if settings_from_file else {})

fix_z = settings['fix_z']
sn_name = settings['sn_name']
selected_bandpasses = settings.get('selected_bandpasses', None)
custom_bandpass_files = settings.get('custom_bandpass_files', None)
svo_filters = settings.get('svo_filters', None) # Get SVO filter definitions

NS_SETTINGS = DEFAULT_NS_SETTINGS.copy()
NS_SETTINGS.update(settings.get('nested_sampling', {}))

PRIOR_BOUNDS = DEFAULT_PRIOR_BOUNDS.copy()
if 'prior_bounds' in settings:
    PRIOR_BOUNDS.update(settings['prior_bounds'])

# Option flag: when fit_sigma is True, an extra parameter is added
fit_sigma = NS_SETTINGS['fit_sigma']
# Flag to determine which model to run primarily (can be overridden)
fit_anomaly = NS_SETTINGS['fit_log_p']

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Define H_ERG_S constant
H_ERG_S = 6.62607015e-27  # Planck constant in erg*s

# Global variable to store band names (will be set in load_and_process_data)
# This approach is fragile; passing unique_bands explicitly is better.
# For now, keeping it to minimize changes to plotting functions.
BAND_NAMES = []

# Load and process data using the updated function from jax_supernovae.data
# This now returns zpbandfluxes as the 7th item
# Note: The custom_load_and_process_data function was removed as its logic
# should now be handled by the updated load_and_process_data and register_all_bandpasses
# If specific filtering or J-band handling is needed, it might need re-integration.
print(f"Loading data for SN: {sn_name}")
# Pass settings for bandpass selection and custom files
# Note: svo_filters are not yet implemented in settings.yaml, passing None for now
times, fluxes, fluxerrs, zps, band_indices, bridges, zpbandfluxes, fixed_z, unique_bands = load_and_process_data(
    sn_name,
    data_dir='hsf_DR1/', # Assuming hsf_DR1 is the correct data dir
    fix_z=fix_z,
    selected_bandpasses=selected_bandpasses, # Pass from settings
    custom_bandpass_files=custom_bandpass_files, # Pass from settings
    svo_filters=svo_filters # Pass SVO filter definitions from settings
)

# unique_bands is now returned directly from load_and_process_data
num_unique_bands = len(unique_bands)
print(f"Using {num_unique_bands} unique bands: {unique_bands}")
BAND_NAMES = unique_bands # Set global for plotting functions

# =============================================================================
# Set up parameter bounds and prior distributions
# =============================================================================

# --- Standard Model Priors ---
standard_param_names = []
standard_prior_dists = {}
standard_param_bounds = {}

if fix_z:
    standard_param_names = ['t0', 'log_x0', 'x1', 'c']
    standard_param_bounds = {
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']), # log_x0 bounds
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    if fit_sigma:
        standard_param_names.append('sigma')
        standard_param_bounds['sigma'] = (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])
else:
    standard_param_names = ['z', 't0', 'log_x0', 'x1', 'c']
    standard_param_bounds = {
        'z': (PRIOR_BOUNDS['z']['min'], PRIOR_BOUNDS['z']['max']),
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']), # log_x0 bounds
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    if fit_sigma:
        standard_param_names.append('sigma')
        standard_param_bounds['sigma'] = (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])

for pname in standard_param_names:
    p_bounds_lookup_key = pname.replace('log_', '') # Key for PRIOR_BOUNDS might not have log_ prefix
    if p_bounds_lookup_key not in PRIOR_BOUNDS:
        raise KeyError(f"Prior bound key '{p_bounds_lookup_key}' (derived from '{pname}') not found in PRIOR_BOUNDS.")
    p_bounds = PRIOR_BOUNDS[p_bounds_lookup_key]
    standard_param_bounds[pname] = (p_bounds['min'], p_bounds['max']) # Store bounds with original name
    standard_prior_dists[pname] = distrax.Uniform(low=p_bounds['min'], high=p_bounds['max']) # Store dist with original name


# --- Anomaly Model Priors ---
anomaly_param_names = []
anomaly_prior_dists = {}
anomaly_param_bounds = {}

if fix_z:
    anomaly_param_names = ['t0', 'log_x0', 'x1', 'c', 'log_p']
    anomaly_param_bounds = {
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']), # log_x0 bounds
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max']),
        'log_p': (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    }
    if fit_sigma:
        anomaly_param_names.insert(4, 'sigma') # Insert sigma before log_p
        anomaly_param_bounds['sigma'] = (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])
else:
    anomaly_param_names = ['z', 't0', 'log_x0', 'x1', 'c', 'log_p']
    anomaly_param_bounds = {
        'z': (PRIOR_BOUNDS['z']['min'], PRIOR_BOUNDS['z']['max']),
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']), # log_x0 bounds
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max']),
        'log_p': (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    }
    if fit_sigma:
        anomaly_param_names.insert(5, 'sigma') # Insert sigma before log_p
        anomaly_param_bounds['sigma'] = (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])

for pname in anomaly_param_names:
    # Determine the key for PRIOR_BOUNDS dictionary
    if pname == 'log_x0':
        p_bounds_lookup_key = 'x0'
    else:
        p_bounds_lookup_key = pname
        
    if p_bounds_lookup_key not in PRIOR_BOUNDS:
         raise KeyError(f"Prior bound key '{p_bounds_lookup_key}' (for parameter '{pname}') not found in PRIOR_BOUNDS.")
    p_bounds = PRIOR_BOUNDS[p_bounds_lookup_key]
    anomaly_param_bounds[pname] = (p_bounds['min'], p_bounds['max']) # Store bounds with original name
    anomaly_prior_dists[pname] = distrax.Uniform(low=p_bounds['min'], high=p_bounds['max']) # Store dist with original name


# =============================================================================
# Log Prior Functions
# =============================================================================
@jax.jit
def logprior_standard(params):
    """Calculate log prior probability for standard nested sampling."""
    logp = 0.0
    for i, pname in enumerate(standard_param_names):
        logp += standard_prior_dists[pname].log_prob(params[i])
    return logp

@jax.jit
def logprior_anomaly(params):
    """Calculate log prior probability for anomaly detection nested sampling."""
    logp = 0.0
    for i, pname in enumerate(anomaly_param_names):
         logp += anomaly_prior_dists[pname].log_prob(params[i])
    return logp

# =============================================================================
# Likelihood Functions (Updated)
# =============================================================================
@jax.jit
def compute_single_loglikelihood_standard(params, zpbandfluxes):
    """Compute Gaussian log likelihood for standard nested sampling."""
    # Ensure params is properly handled for both single and batched inputs
    params = jnp.atleast_1d(params) # Use atleast_1d based on ns.py fix
    if params.ndim > 1:
        # If we have a batch, vmap over it
        # Pass zpbandfluxes to the vmapped function using a lambda
        return jax.vmap(lambda p: compute_single_loglikelihood_standard(p, zpbandfluxes))(params)

    # Unpack parameters based on whether sigma and z are fixed
    param_dict_local = {}
    current_idx = 0
    if not fix_z:
        param_dict_local['z'] = params[current_idx]
        current_idx += 1
    else:
        param_dict_local['z'] = fixed_z[0]

    param_dict_local['t0'] = params[current_idx]
    current_idx += 1
    param_dict_local['x0'] = 10**params[current_idx] # log_x0 -> x0
    current_idx += 1
    param_dict_local['x1'] = params[current_idx]
    current_idx += 1
    param_dict_local['c'] = params[current_idx]
    current_idx += 1

    if fit_sigma:
        sigma = params[current_idx]
    else:
        sigma = 1.0

    # Calculate model fluxes (unscaled) for all observations at once
    model_fluxes_unscaled_allbands = optimized_salt3_multiband_flux(times, bridges, param_dict_local)
    # Select the flux for the correct band for each time point
    model_fluxes_unscaled = model_fluxes_unscaled_allbands[jnp.arange(len(times)), band_indices]

    # Apply zero-point scaling using precomputed zpbandfluxes
    zpbf_per_time = zpbandfluxes[band_indices]
    zpnorm_per_time = jnp.where(zpbf_per_time > 1e-30, 10**(0.4 * zps) / zpbf_per_time, 0.0)
    model_fluxes_scaled = model_fluxes_unscaled * zpnorm_per_time

    # Calculate likelihood using scaled fluxes
    eff_fluxerrs = sigma * fluxerrs
    chi2 = jnp.sum(((fluxes - model_fluxes_scaled) / eff_fluxerrs) ** 2)
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * eff_fluxerrs ** 2)))
    return log_likelihood

@jax.jit
def compute_single_loglikelihood_anomaly(params, zpbandfluxes):
    """Compute Gaussian log likelihood for a single set of parameters with anomaly detection."""
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        return jax.vmap(lambda p: compute_single_loglikelihood_anomaly(p, zpbandfluxes))(params)

    # Unpack parameters
    param_dict_local = {}
    current_idx = 0
    if not fix_z:
        param_dict_local['z'] = params[current_idx]
        current_idx += 1
    else:
        param_dict_local['z'] = fixed_z[0]

    param_dict_local['t0'] = params[current_idx]
    current_idx += 1
    param_dict_local['x0'] = 10**params[current_idx] # log_x0 -> x0
    current_idx += 1
    param_dict_local['x1'] = params[current_idx]
    current_idx += 1
    param_dict_local['c'] = params[current_idx]
    current_idx += 1

    if fit_sigma:
        sigma = params[current_idx]
        current_idx += 1
    else:
        sigma = 1.0

    log_p = params[current_idx] # Last param is log_p
    p = jnp.exp(log_p)
    p = jnp.clip(p, 1e-9, 1.0 - 1e-9)

    # Calculate model fluxes (unscaled)
    model_fluxes_unscaled_allbands = optimized_salt3_multiband_flux(times, bridges, param_dict_local)
    model_fluxes_unscaled = model_fluxes_unscaled_allbands[jnp.arange(len(times)), band_indices]

    # Apply zero-point scaling
    zpbf_per_time = zpbandfluxes[band_indices]
    zpnorm_per_time = jnp.where(zpbf_per_time > 1e-30, 10**(0.4 * zps) / zpbf_per_time, 0.0)
    model_fluxes_scaled = model_fluxes_unscaled * zpnorm_per_time

    # Calculate likelihood using scaled fluxes, incorporating anomaly probability
    eff_fluxerrs = sigma * fluxerrs

    # Standard Gaussian likelihood component
    chi2_standard = ((fluxes - model_fluxes_scaled) / eff_fluxerrs) ** 2
    log_likelihood_standard = -0.5 * (chi2_standard + jnp.log(2 * jnp.pi * eff_fluxerrs ** 2))

    # Outlier likelihood component (broad Gaussian)
    flux_range = jnp.max(fluxes) - jnp.min(fluxes)
    outlier_variance = (flux_range)**2
    outlier_variance = jnp.maximum(outlier_variance, 1e-9)
    chi2_outlier = ((fluxes - model_fluxes_scaled)**2) / outlier_variance
    log_likelihood_outlier = -0.5 * (chi2_outlier + jnp.log(2 * jnp.pi * outlier_variance))

    # Combine standard and outlier likelihoods
    log_likelihood_combined = jnp.logaddexp(
        jnp.log(1 - p) + log_likelihood_standard,
        jnp.log(p) + log_likelihood_outlier
    )

    total_log_likelihood = jnp.sum(log_likelihood_combined)
    return total_log_likelihood

# =============================================================================
# Function to sample from the prior distributions.
# =============================================================================
def sample_from_priors(rng_key, n_samples, param_names, prior_dists):
    """Samples from the prior distributions for the given parameter names."""
    keys = jax.random.split(rng_key, len(param_names))
    samples = [prior_dists[param].sample(seed=k, sample_shape=(n_samples,))
               for param, k in zip(param_names, keys)]
    return jnp.column_stack(samples)

# =============================================================================
# Function to run nested sampling.
# =============================================================================
def run_nested_sampling(ll_fn, logprior_fn, param_names, prior_dists, output_prefix, sn_name, identifier="", num_iterations=NS_SETTINGS['max_iterations']):
    """Run nested sampling with output directories/files including supernova name."""

    output_dir = os.path.join("results", f"chains_{sn_name}{identifier}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running {output_prefix} nested sampling for {output_dir}...")

    n_params = len(param_names)
    num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']

    # Create partial function for the likelihood with zpbandfluxes bound
    # Note: zpbandfluxes is available in the global scope where run_nested_sampling is called
    loglikelihood_partial = partial(ll_fn, zpbandfluxes=zpbandfluxes)

    # Initialize nested sampling algorithm
    algo = blackjax.ns.adaptive.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_partial, # Use the partial function
        n_delete=NS_SETTINGS['n_delete'],
        num_mcmc_steps=num_mcmc_steps,
    )

    # Initialize random key and particles
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)

    # Generate initial particles from prior
    initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'], param_names, prior_dists)
    print("Initial particles generated, shape: ", initial_particles.shape)

    # Initialize state using the partial likelihood function
    # Blackjax init expects a function that takes only params
    state = algo.init(initial_particles, loglikelihood_partial)

    # Define one_step function with JIT
    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        # algo.step uses the loglikelihood_fn provided during algo creation (the partial one)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    # Run nested sampling
    dead = []

    print("Running nested sampling...")
    with tqdm.tqdm(desc="Dead points", unit=" dead points", total=num_iterations*NS_SETTINGS['n_delete']) as pbar:
        for i in range(num_iterations):
            if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
                break
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(NS_SETTINGS['n_delete'])
            if i % 10 == 0:
                print(f"Iteration {i}: logZ = {state.sampler_state.logZ:.2f}")

    dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
    logw = log_weights(rng_key, dead)
    logZs = jax.scipy.special.logsumexp(logw, axis=0)
    print(f"Runtime evidence: {state.sampler_state.logZ:.2f}")
    print(f"Estimated evidence: {logZs.mean():.2f} +- {logZs.std():.2f}")

    # Save chains with the correct filename
    chains_filename = f"{output_prefix}_dead-birth.txt"
    final_path = os.path.join(output_dir, chains_filename)

    # Extract data from dead info
    points = np.array(dead.particles)
    logls_death = np.array(dead.logL)
    logls_birth = np.array(dead.logL_birth)

    # Combine data: parameters, death likelihood, birth likelihood
    data_to_save = np.column_stack([points, logls_death, logls_birth])

    # Save directly to final location
    np.savetxt(final_path, data_to_save)
    print(f"Saved {data_to_save.shape[0]} samples to {final_path}")

    # Return samples for plotting by reading the saved file
    # anesthetic expects the root path for the chains file
    return read_chains(final_path, columns=param_names)


# =============================================================================
# Helper functions for plotting and analysis (modified)
# =============================================================================

def get_n_params(param_names):
    """Get the number of parameters being fit."""
    return len(param_names)

def get_true_values(sn_name, data_dir='hsf_DR1/', selected_bandpasses=None):
    """
    Read the true values from the salt_fits.dat file for a given supernova.
    Only returns values for exactly matching bandpass combinations.
    """
    if selected_bandpasses is None:
        print("Warning: No selected_bandpasses provided to get_true_values. Cannot find true values.")
        return None

    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    salt_fits_path = os.path.join(project_root_dir, data_dir, 'Ia', sn_name, 'salt_fits.dat')

    if not os.path.exists(salt_fits_path):
        print(f"Warning: salt_fits.dat not found at {salt_fits_path}")
        return None

    try:
        with open(salt_fits_path, 'r') as f: lines = f.readlines()
        header = lines[0].strip().split()
        selected_bandpasses_sorted = sorted(selected_bandpasses)
        target_bps = '-'.join(selected_bandpasses_sorted)

        matching_row = None
        for line in lines[1:]:
            values = line.strip().split()
            if not values: continue
            # Sort the bands listed in the file's first column
            file_bps_sorted = '-'.join(sorted(values[0].split('-')))
            if file_bps_sorted == target_bps:
                matching_row = values
                break

        if matching_row is None:
            print(f"Warning: No matching band combination '{target_bps}' found in {salt_fits_path}")
            return None

        try:
            t0_idx = header.index('t0')
            x0_mag_idx = header.index('x0_mag')
            x1_idx = header.index('x1')
            c_idx = header.index('c')
            t0 = float(matching_row[t0_idx])
            x0_mag = float(matching_row[x0_mag_idx])
            x1 = float(matching_row[x1_idx])
            c = float(matching_row[c_idx])
        except (ValueError, IndexError) as e:
             print(f"Error parsing matching row in {salt_fits_path}: {e}")
             return None

        true_values = {'t0': t0, 'log_x0': -x0_mag / 2.5, 'x1': x1, 'c': c}
        return true_values

    except Exception as e:
        print(f"\nError in get_true_values: {str(e)}")
        return None

def plot_x0mag_dm_relationship(output_dir):
    """ Create a plot showing the relationship between x0_mag and DM. """
    try:
        x0_mag_values = np.linspace(8, 10, 100)
        dm_values = x0_mag_values + 21.01 # Assuming fixed relation
        plt.figure(figsize=(10, 6))
        plt.plot(x0_mag_values, dm_values, 'b-', linewidth=2)
        plt.xlabel('x0_mag = -2.5 * log10(x0)', fontsize=12)
        plt.ylabel('Distance Modulus (DM)', fontsize=12)
        plt.title('Relationship between x0_mag and Distance Modulus', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.text(0.05, 0.95, 'DM = x0_mag + 21.01', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'x0mag_dm_relationship.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved x0_mag vs DM relationship plot to {os.path.join(output_dir, 'x0mag_dm_relationship.png')}")
    except Exception as e:
        print(f"Warning: Failed to create x0_mag vs DM relationship plot - {str(e)}")
        plt.close()

def plot_samples_x0mag_dm(standard_samples, anomaly_samples, true_values, output_dir):
    """ Create a scatter plot of the samples showing x0_mag vs DM. """
    try:
        has_std_x0 = standard_samples is not None and 'log_x0' in standard_samples.columns
        has_anom_x0 = anomaly_samples is not None and 'log_x0' in anomaly_samples.columns
        if not has_std_x0 and not has_anom_x0:
            print("Warning: log_x0 not found in samples - skipping x0_mag vs DM scatter plot")
            return

        plt.figure(figsize=(10, 8))
        x0_mag_std, dm_std = (None, None)
        x0_mag_anom, dm_anom = (None, None)

        if has_std_x0:
            x0_mag_std = -2.5 * standard_samples['log_x0']
            dm_std = x0_mag_std + 21.01
            plt.scatter(x0_mag_std, dm_std, alpha=0.5, label='Standard', s=10)

        if has_anom_x0:
            x0_mag_anom = -2.5 * anomaly_samples['log_x0']
            dm_anom = x0_mag_anom + 21.01
            plt.scatter(x0_mag_anom, dm_anom, alpha=0.5, label='Anomaly', s=10)

        if true_values and 'log_x0' in true_values:
            true_x0_mag = -2.5 * true_values['log_x0']
            true_dm = true_x0_mag + 21.01
            plt.scatter([true_x0_mag], [true_dm], color='red', marker='*', s=200, label='True Value', zorder=10)

        # Determine plot limits based on available data
        all_x0_mags = []
        if x0_mag_std is not None: all_x0_mags.extend(x0_mag_std)
        if x0_mag_anom is not None: all_x0_mags.extend(x0_mag_anom)
        if all_x0_mags:
             x_min, x_max = np.min(all_x0_mags), np.max(all_x0_mags)
             x_pad = (x_max - x_min) * 0.05 # Add padding
             x_vals = np.linspace(x_min - x_pad, x_max + x_pad, 100)
             plt.plot(x_vals, x_vals + 21.01, 'k--', label='DM = x0_mag + 21.01')
             plt.xlim(x_min - x_pad, x_max + x_pad) # Set xlim based on data
             plt.ylim(x_min + 21.01 - x_pad, x_max + 21.01 + x_pad) # Set ylim based on data

        plt.xlabel('x0_mag = -2.5 * log10(x0)', fontsize=12)
        plt.ylabel('Distance Modulus (DM = x0_mag + 21.01)', fontsize=12)
        plt.title('Relationship between x0_mag and Distance Modulus in Samples', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'samples_x0mag_dm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved samples x0_mag vs DM plot to {os.path.join(output_dir, 'samples_x0mag_dm.png')}")

    except Exception as e:
        print(f"Warning: Failed to create samples x0_mag vs DM plot - {str(e)}")
        plt.close()

def get_model_curve(samples, param_names_for_model):
    """Get model curve for median parameters."""
    params = {}
    # Use median (50th percentile) for robustness
    percentile = 50

    # Ensure samples is a DataFrame if it's anesthetic object
    if hasattr(samples, 'dataframe'):
        samples_df = samples.dataframe
    else:
        samples_df = samples # Assume it's already a DataFrame or similar

    for param in param_names_for_model:
        if param in samples_df.columns:
             params[param] = float(np.percentile(samples_df[param], percentile))
        else:
             print(f"Warning: Parameter '{param}' not found in samples for get_model_curve.")
             continue

    # Convert log_x0 to x0 if present
    if 'log_x0' in params:
        params['x0'] = 10**params['log_x0']
        del params['log_x0']

    # Ensure required base parameters are present or defaulted
    if fix_z:
        params['z'] = fixed_z[0]
    elif 'z' not in params:
         raise ValueError("Redshift 'z' missing from parameters and not fixed.")

    if 't0' not in params: raise ValueError("'t0' missing from parameters.")
    if 'x0' not in params: raise ValueError("'x0' (derived from log_x0) missing from parameters.")
    if 'x1' not in params: raise ValueError("'x1' missing from parameters.")
    if 'c' not in params: raise ValueError("'c' missing from parameters.")

    # Remove log_p if it accidentally got included
    if 'log_p' in params: del params['log_p']
    # Handle sigma
    if 'sigma' not in params and not fit_sigma:
        params['sigma'] = 1.0
    elif 'sigma' not in params and fit_sigma:
        print("Warning: fit_sigma is True but 'sigma' not found in median params. Using default 1.0.")
        params['sigma'] = 1.0

    return params

# =============================================================================
# Main execution block
# =============================================================================
if __name__ == "__main__":
    # Add an identifier for this run (e.g. date, version, etc)
    identifier = "_update_test" # Changed identifier

    # --- Run Anomaly Model (if selected) ---
    anomaly_samples = None
    if fit_anomaly:
        print("\nRunning anomaly detection version...")
        anomaly_samples = run_nested_sampling(
            ll_fn=compute_single_loglikelihood_anomaly, # Pass original function object
            logprior_fn=logprior_anomaly,
            param_names=anomaly_param_names,
            prior_dists=anomaly_prior_dists,
            output_prefix="chains_anomaly",
            sn_name=sn_name,
            identifier=identifier
        )

    # --- Run Standard Model ---
    print("\nRunning standard version...")
    standard_samples = run_nested_sampling(
        ll_fn=compute_single_loglikelihood_standard, # Pass original function object
        logprior_fn=logprior_standard,
        param_names=standard_param_names,
        prior_dists=standard_prior_dists,
        output_prefix="chains_standard",
        sn_name=sn_name,
        identifier=identifier
    )

    # --- Plotting and Analysis ---
    print("\nGenerating plots...")
    output_dir = f'results/chains_{sn_name}{identifier}'
    os.makedirs(output_dir, exist_ok=True)

    # Get true values if possible
    true_values = get_true_values(sn_name, selected_bandpasses=unique_bands)
    print(f"True values from salt_fits.dat: {true_values}")

    # Check if samples were loaded correctly
    have_standard = standard_samples is not None
    have_anomaly = anomaly_samples is not None

    # Parameter names for plotting (common base parameters)
    plot_param_names = standard_param_names if have_standard else anomaly_param_names
    # Remove parameters not common or not desired in comparison plot
    params_to_remove_from_plot = ['log_p']
    if not fit_sigma: params_to_remove_from_plot.append('sigma')
    plot_param_names = [p for p in plot_param_names if p not in params_to_remove_from_plot]


    # Only create corner plot if we have at least one set of chains
    if have_standard or have_anomaly:
        try:
            plot_param_names_for_corner = plot_param_names.copy()
            plot_true_values_for_corner = true_values.copy() if true_values else {}
            standard_plot_samples = None
            anomaly_plot_samples_for_comp = None # Use a different var for the comparison plot samples

            # Convert log_x0 to x0_mag for plotting
            if 'log_x0' in plot_param_names_for_corner:
                x0_mag_idx = plot_param_names_for_corner.index('log_x0')
                plot_param_names_for_corner[x0_mag_idx] = 'x0_mag'
                if plot_true_values_for_corner and 'log_x0' in plot_true_values_for_corner:
                    plot_true_values_for_corner['x0_mag'] = -2.5 * plot_true_values_for_corner['log_x0']
                    del plot_true_values_for_corner['log_x0']
                if have_standard:
                    standard_plot_samples = standard_samples.copy()
                    standard_plot_samples['x0_mag'] = -2.5 * standard_plot_samples['log_x0']
                    standard_plot_samples = standard_plot_samples.drop(columns=['log_x0'])
                if have_anomaly:
                    # Select only common columns for comparison plot
                    anomaly_plot_samples_for_comp = anomaly_samples.copy()
                    anomaly_plot_samples_for_comp['x0_mag'] = -2.5 * anomaly_plot_samples_for_comp['log_x0']
                    # Drop columns not in plot_param_names_for_corner (like log_p, maybe sigma)
                    cols_to_drop = [c for c in anomaly_plot_samples_for_comp.columns if c not in plot_param_names_for_corner and c != 'x0_mag']
                    if 'log_x0' in cols_to_drop: cols_to_drop.remove('log_x0') # Keep log_x0 if needed for x0_mag calc
                    anomaly_plot_samples_for_comp = anomaly_plot_samples_for_comp.drop(columns=cols_to_drop)

            else: # If no log_x0, use original samples
                 if have_standard: standard_plot_samples = standard_samples
                 if have_anomaly: 
                     anomaly_plot_samples_for_comp = anomaly_samples.copy()
                     cols_to_drop = [c for c in anomaly_plot_samples_for_comp.columns if c not in plot_param_names_for_corner]
                     anomaly_plot_samples_for_comp = anomaly_plot_samples_for_comp.drop(columns=cols_to_drop)


            fig, axes = make_2d_axes(plot_param_names_for_corner, figsize=(10, 10), facecolor='w')

            if have_standard: standard_plot_samples.plot_2d(axes, alpha=0.7, label="Standard")
            if have_anomaly: anomaly_plot_samples_for_comp.plot_2d(axes, alpha=0.7, label="Anomaly") # Plot modified samples

            if plot_true_values_for_corner:
                 axes.axlines(plot_true_values_for_corner, c='green', linestyle='--', linewidth=2, alpha=1.0, zorder=10, label='True values')

            handles, labels = axes.iloc[-1, 0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes.iloc[-1, 0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=3)

            plt.savefig(f'{output_dir}/corner_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to create corner comparison plot - {str(e)}")
            if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

    # Create anomaly corner plot only if anomaly run was successful
    if have_anomaly and 'log_p' in anomaly_samples.columns:
        try:
            anom_plot_params = anomaly_param_names.copy()
            anom_plot_true = true_values.copy() if true_values else {}
            anom_plot_samples = anomaly_samples.copy() # Use original anomaly samples

            if 'log_x0' in anom_plot_params:
                anom_plot_params[anom_plot_params.index('log_x0')] = 'x0_mag'
                if anom_plot_true and 'log_x0' in anom_plot_true:
                    anom_plot_true['x0_mag'] = -2.5 * anom_plot_true['log_x0']
                    del anom_plot_true['log_x0']
                anom_plot_samples['x0_mag'] = -2.5 * anom_plot_samples['log_x0']
                anom_plot_samples = anom_plot_samples.drop(columns=['log_x0'])

            fig_anom, axes_anom = make_2d_axes(anom_plot_params, figsize=(12, 12), facecolor='w')
            anom_plot_samples.plot_2d(axes_anom, alpha=0.7, label="Anomaly")

            if anom_plot_true:
                 axes_anom.axlines(anom_plot_true, c='green', linestyle='--', linewidth=2, alpha=1.0, zorder=10, label='True values')

            plt.suptitle('Anomaly Detection Corner Plot', fontsize=14)
            handles, labels = axes_anom.iloc[-1, 0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes_anom.iloc[-1, 0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(len(axes_anom)/2, len(axes_anom)), loc='lower center', ncol=2)

            plt.savefig(f'{output_dir}/corner_anomaly_logp.png', dpi=300, bbox_inches='tight')
            plt.close(fig_anom)
        except Exception as e:
            print(f"Warning: Failed to create anomaly corner plot - {str(e)}")
            if 'fig_anom' in locals() and plt.fignum_exists(fig_anom.number): plt.close(fig_anom)

    # Create Light Curve Plot
    try:
        t_min = np.min(times) - 5
        t_max = np.max(times) + 5
        t_grid = np.linspace(t_min, t_max, 100)
        n_bands_plot = len(unique_bands) # Use the reconstructed unique_bands

        fig_lc = plt.figure(figsize=(15, 8)) # Adjusted size
        ax_lc = fig_lc.add_subplot(111)

        default_colours = plt.cm.tab10.colors
        default_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '*']
        colours = [default_colours[i % len(default_colours)] for i in range(n_bands_plot)]
        markers = [default_markers[i % len(default_markers)] for i in range(n_bands_plot)]

        # Plot data points
        for i, band_name in enumerate(unique_bands):
             mask = band_indices == i
             if not np.any(mask): continue
             ax_lc.errorbar(times[mask], fluxes[mask], yerr=fluxerrs[mask],
                           fmt=markers[i], color=colours[i], label=f'{band_name} Data',
                           markersize=8, alpha=0.6)

        # Plot model curves
        if have_standard:
            try:
                params_std = get_model_curve(standard_samples, standard_param_names)
                model_fluxes_std_unscaled = optimized_salt3_multiband_flux(jnp.array(t_grid), bridges, params_std)
                # Apply ZP scaling for the grid
                zpbf_grid = zpbandfluxes # Assuming zpbandfluxes corresponds to bridges order
                # Use mean zp for scaling the grid curve - this might not be ideal
                mean_zp = np.mean(zps)
                zpnorm_grid = jnp.where(zpbf_grid > 1e-30, 10**(0.4 * mean_zp) / zpbf_grid, 0.0)
                model_fluxes_std_scaled = model_fluxes_std_unscaled * zpnorm_grid[None, :]
                for i, band_name in enumerate(unique_bands):
                    ax_lc.plot(t_grid, model_fluxes_std_scaled[:, i], '--', color=colours[i], label=f'{band_name} Standard Fit', linewidth=2, alpha=0.8)
            except Exception as e:
                 print(f"Warning: Failed to plot standard model curve - {str(e)}")

        if have_anomaly:
             try:
                params_anom = get_model_curve(anomaly_samples, anomaly_param_names)
                model_fluxes_anom_unscaled = optimized_salt3_multiband_flux(jnp.array(t_grid), bridges, params_anom)
                zpbf_grid = zpbandfluxes
                mean_zp = np.mean(zps)
                zpnorm_grid = jnp.where(zpbf_grid > 1e-30, 10**(0.4 * mean_zp) / zpbf_grid, 0.0)
                model_fluxes_anom_scaled = model_fluxes_anom_unscaled * zpnorm_grid[None, :]
                for i, band_name in enumerate(unique_bands):
                    ax_lc.plot(t_grid, model_fluxes_anom_scaled[:, i], '-', color=colours[i], label=f'{band_name} Anomaly Fit', linewidth=2, alpha=0.8)
             except Exception as e:
                 print(f"Warning: Failed to plot anomaly model curve - {str(e)}")


        ax_lc.set_xlabel('MJD', fontsize=12)
        ax_lc.set_ylabel('Flux (Scaled by Mean ZP)', fontsize=12) # Indicate scaling method
        title_lc = f'Light Curve Fit Comparison for {sn_name}'
        if fix_z and fixed_z is not None: title_lc += f' (z = {fixed_z[0]:.4f})'
        ax_lc.set_title(title_lc, fontsize=14)

        # Consolidate legend
        handles_lc, labels_lc = ax_lc.get_legend_handles_labels()
        by_label_lc = dict(zip(labels_lc, handles_lc))
        ax_lc.legend(by_label_lc.values(), by_label_lc.keys(), ncol=2, fontsize=10)

        ax_lc.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/light_curve_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig_lc)

    except Exception as e:
        print(f"Warning: Failed to create light curve comparison plot - {str(e)}")
        if 'fig_lc' in locals() and plt.fignum_exists(fig_lc.number): plt.close(fig_lc)

    # Save parameter statistics
    if have_standard or have_anomaly:
        stats_text = [f"Parameter Statistics Comparison for {sn_name}:", "-" * 50]
        all_params_for_stats = sorted(list(set(standard_param_names + anomaly_param_names)))

        for param in all_params_for_stats:
            stats_text.append(f"\n{param}:")
            if have_standard and param in standard_samples.columns:
                std_mean = standard_samples[param].mean()
                std_std = standard_samples[param].std()
                stats_text.append(f"  Standard: {std_mean:.6f} ± {std_std:.6f}")
            if have_anomaly and param in anomaly_samples.columns:
                anom_mean = anomaly_samples[param].mean()
                anom_std = anomaly_samples[param].std()
                stats_text.append(f"  Anomaly:  {anom_mean:.6f} ± {anom_std:.6f}")

        if 'log_x0' in all_params_for_stats:
            stats_text.append(f"\nx0_mag (calculated from log_x0):")
            x0_mag_std, x0_mag_anom = None, None # Initialize
            if have_standard and 'log_x0' in standard_samples.columns:
                x0_mag_std = -2.5 * standard_samples['log_x0']
                stats_text.append(f"  Standard: {x0_mag_std.mean():.6f} ± {x0_mag_std.std():.6f}")
            if have_anomaly and 'log_x0' in anomaly_samples.columns:
                x0_mag_anom = -2.5 * anomaly_samples['log_x0']
                stats_text.append(f"  Anomaly:  {x0_mag_anom.mean():.6f} ± {x0_mag_anom.std():.6f}")

            stats_text.append(f"\nDistance Modulus (DM = x0_mag + 21.01):")
            if x0_mag_std is not None:
                dm_std = x0_mag_std + 21.01
                stats_text.append(f"  Standard: {dm_std.mean():.6f} ± {dm_std.std():.6f}")
            if x0_mag_anom is not None:
                dm_anom = x0_mag_anom + 21.01
                stats_text.append(f"  Anomaly:  {dm_anom.mean():.6f} ± {dm_anom.std():.6f}")

        stats_text = '\n'.join(stats_text)
        with open(f'{output_dir}/parameter_statistics.txt', 'w') as f: f.write(stats_text)
        print(f"\nParameter statistics saved to {output_dir}/parameter_statistics.txt")

    # Create relationship plots
    plot_x0mag_dm_relationship(output_dir)
    plot_samples_x0mag_dm(standard_samples, anomaly_samples, true_values, output_dir)