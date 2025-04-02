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
import yaml # Added import
from functools import partial
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3 import (
    optimized_salt3_multiband_flux,
)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".2"

from jax_supernovae.bandpasses import register_bandpass, get_bandpass, register_all_bandpasses
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_and_process_data
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# --- Load Settings ---
DEFAULT_NS_SETTINGS_NS = {
    'n_delete': 1,
    'n_live': 125,
    'num_mcmc_steps_multiplier': 5,
    'fit_sigma': False, # Default for this script
    'fix_z': True # Default for this script
}

DEFAULT_PRIOR_BOUNDS_NS = {
    'z': {'min': 0.001, 'max': 0.2},
    't0': {'min': 58000.0, 'max': 59000.0},
    'x0': {'min': -5.0, 'max': -2.6}, # Note: This is log10(x0)
    'x1': {'min': -4.0, 'max': 4.0},
    'c': {'min': -0.3, 'max': 0.3},
    'log_sigma': {'min': -3.0, 'max': 1.0}
}

DEFAULT_SETTINGS_NS = {
    'sn_name': '19dwz', # Default SN for this script
    'selected_bandpasses': None,
    'custom_bandpass_files': None,
    'svo_filters': None
}

# Try to load settings.yaml
try:
    with open('settings.yaml', 'r') as f:
        settings_from_file = yaml.safe_load(f)
except FileNotFoundError:
    settings_from_file = {}

# Merge settings, prioritizing file settings over defaults
settings = DEFAULT_SETTINGS_NS.copy()
settings.update(settings_from_file if settings_from_file else {})

NS_SETTINGS = DEFAULT_NS_SETTINGS_NS.copy()
NS_SETTINGS.update(settings.get('nested_sampling', {}))

PRIOR_BOUNDS = DEFAULT_PRIOR_BOUNDS_NS.copy()
if 'prior_bounds' in settings:
    PRIOR_BOUNDS.update(settings['prior_bounds'])

# Extract settings used by this script
sn_name = settings['sn_name']
fix_z = NS_SETTINGS['fix_z']
fit_sigma = NS_SETTINGS['fit_sigma'] # Allow override from settings
selected_bandpasses = settings.get('selected_bandpasses', None)
custom_bandpass_files = settings.get('custom_bandpass_files', None)
svo_filters = settings.get('svo_filters', None) # Get SVO filters

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load and process data, passing bandpass settings and unpacking unique_bands
times, fluxes, fluxerrs, zps, band_indices, bridges, zpbandfluxes, fixed_z, unique_bands = load_and_process_data(
    sn_name,
    data_dir='hsf_DR1/', # Use the correct data directory
    fix_z=fix_z,
    selected_bandpasses=selected_bandpasses,
    custom_bandpass_files=custom_bandpass_files,
    svo_filters=svo_filters
)
print(f"Using {len(unique_bands)} unique bands for SN {sn_name}: {unique_bands}")

# Define parameter bounds and priors dynamically
param_names_base = []
if not fix_z:
    param_names_base.append('z')
# Use 'x0' for PRIOR_BOUNDS lookup, will rename to 'log_x0' later for internal use
param_names_base.extend(['t0', 'x0', 'x1', 'c'])
if fit_sigma:
    param_names_base.append('log_sigma')

param_bounds = {}
prior_dists = {}
for pname in param_names_base:
    # Use the name directly for PRIOR_BOUNDS lookup
    if pname not in PRIOR_BOUNDS:
         raise KeyError(f"Prior bound key '{pname}' not found in PRIOR_BOUNDS.")
    p_bounds = PRIOR_BOUNDS[pname]
    param_bounds[pname] = (p_bounds['min'], p_bounds['max'])
    prior_dists[pname] = distrax.Uniform(low=p_bounds['min'], high=p_bounds['max'])

# Define the list of parameter names used internally (with log_x0)
param_names = param_names_base.copy()
if 'x0' in param_names:
    param_names[param_names.index('x0')] = 'log_x0'

print(f"Fitting parameters: {param_names}")

# Define logprior based on dynamic param_names and prior_dists
@jax.jit
def logprior(params):
    """Calculate log prior probability."""
    logp = 0.0
    # params array order matches param_names list order
    for i, pname in enumerate(param_names):
        # Need to look up the correct prior dist (use base name 'x0' if internal name is 'log_x0')
        prior_lookup_name = 'x0' if pname == 'log_x0' else pname
        logp += prior_dists[prior_lookup_name].log_prob(params[i])
    return logp

# Define H_ERG_S constant if not already available globally
H_ERG_S = 6.62607015e-27  # Planck constant in erg*s

# zpbandfluxes array (shape n_bands) is now loaded by load_and_process_data

@jax.jit
def compute_single_loglikelihood(params, zpbandfluxes):
    """Compute Gaussian log likelihood for a single set of parameters."""
    # Ensure params is properly handled for both single and batched inputs
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        # If we have a batch, vmap over it
        return jax.vmap(lambda p: compute_single_loglikelihood(p, zpbandfluxes))(params)

    # Unpack parameters dynamically based on param_names
    param_dict_local = {}
    current_idx = 0
    sigma = 1.0 # Default sigma

    if not fix_z:
        param_dict_local['z'] = params[current_idx]
        current_idx += 1
    else:
        param_dict_local['z'] = fixed_z[0] # Use fixed redshift if applicable

    # Expect t0, log_x0, x1, c in param_names order
    # Map params array index to parameter name
    param_map = {name: params[i] for i, name in enumerate(param_names)}

    param_dict_local['t0'] = param_map['t0']
    param_dict_local['x0'] = 10**param_map['log_x0'] # Convert log_x0 to x0
    param_dict_local['x1'] = param_map['x1']
    param_dict_local['c'] = param_map['c']

    if fit_sigma:
        log_sigma = param_map['log_sigma']
        sigma = 10**log_sigma

    # Calculate model fluxes (unscaled) for all observations at once
    model_fluxes_unscaled_allbands = optimized_salt3_multiband_flux(times, bridges, param_dict_local)
    # Select the flux for the correct band for each time point
    model_fluxes_unscaled = model_fluxes_unscaled_allbands[jnp.arange(len(times)), band_indices]

    # Apply zero-point scaling using precomputed zpbandfluxes
    # Get zpbandflux for each time point based on its band index
    # zpbandfluxes (shape n_bands) is passed as an argument
    zpbf_per_time = zpbandfluxes[band_indices]
    # Calculate normalization factor for each time point using its zp
    zpnorm_per_time = jnp.where(zpbf_per_time > 1e-30, 10**(0.4 * zps) / zpbf_per_time, 0.0)
    # Apply scaling
    model_fluxes_scaled = model_fluxes_unscaled * zpnorm_per_time

    # Calculate likelihood using scaled fluxes
    eff_fluxerrs = sigma * fluxerrs  # effective flux errors
    chi2 = jnp.sum(((fluxes - model_fluxes_scaled) / eff_fluxerrs) ** 2)
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * eff_fluxerrs ** 2)))
    return log_likelihood

def sample_from_priors(rng_key, n_samples):
    """Sample from all prior distributions based on param_names."""
    # Use the dynamically determined param_names list
    keys = jax.random.split(rng_key, len(param_names))
    samples = []
    for i, pname in enumerate(param_names):
         # Need to look up the correct prior dist (use base name 'x0' if internal name is 'log_x0')
        prior_lookup_name = 'x0' if pname == 'log_x0' else pname
        samples.append(prior_dists[prior_lookup_name].sample(seed=keys[i], sample_shape=(n_samples,)))
    return jnp.column_stack(samples)

# Use the length of the dynamic param_names list
n_params_total = len(param_names)
num_mcmc_steps = n_params_total * NS_SETTINGS['num_mcmc_steps_multiplier']

# Create a partial function for the likelihood with zpbandfluxes bound
loglikelihood_for_blackjax = partial(compute_single_loglikelihood, zpbandfluxes=zpbandfluxes)

# Initialize nested sampling algorithm
print("Setting up nested sampling algorithm...")
algo = blackjax.ns.adaptive.nss(
    logprior_fn=logprior,
    loglikelihood_fn=loglikelihood_for_blackjax,
    n_delete=NS_SETTINGS['n_delete'],
    num_mcmc_steps=num_mcmc_steps,
)

# Initialize random key and particles
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key)

initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'])
print("Initial particles generated, shape: ", initial_particles.shape)

# Initialize state
state = algo.init(initial_particles, loglikelihood_for_blackjax)

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

# Save chains using the utility function - param_names is already defined dynamically
save_chains_dead_birth(dead, param_names, sn_name=sn_name) # Pass sn_name for unique filename

# Read the chains and create visualizations
print("\nCreating corner plot...")
# Use sn_name in the path
samples = read_chains(f'chains/chains_{sn_name}', columns=param_names)

# Create corner plot with improved styling
fig, axes = make_2d_axes(param_names, figsize=(12, 12), facecolor='w')
samples.plot_2d(axes, alpha=0.9, label="posterior")

# Improve plot aesthetics
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)),
                       loc='lower center', ncols=2)
plt.suptitle('SALT3 Parameter Posterior Distributions', y=1.02, fontsize=14)

# Save the plot with high DPI and unique name
plot_filename = f'corner_plot_{sn_name}.png'
print(f"Saving corner plot to '{plot_filename}'...")
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
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

# Optional: Save parameter statistics to a file with unique name
stats_filename = f'parameter_statistics_{sn_name}.txt'
with open(stats_filename, 'w') as f:
    f.write(f"Parameter Statistics for {sn_name}:\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Parameter':<12} {'Mean':>15} {'Std Dev':>15}\n")
    f.write("-" * 50 + "\n")
    for param in param_names:
        mean = samples[param].mean()
        std = samples[param].std()
        f.write(f"{param:<12} {mean:>15.6f} {std:>15.6f}\n")
    f.write("-" * 50 + "\n")
print(f"\nParameter statistics saved to '{stats_filename}'")