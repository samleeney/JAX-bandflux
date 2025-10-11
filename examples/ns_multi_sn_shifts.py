"""
# Multi-Supernova Nested Sampling with Shared Transmission Shifts

This script demonstrates how to fit multiple supernovae simultaneously with:
- Individual SALT parameters (x0, x1, t0, c) for each supernova
- Shared transmission shift parameters across all supernovae for each filter

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
from jax_supernovae.bandpasses import register_bandpass, get_bandpass
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_multiple_supernovae
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# Configuration
SN_NAMES = ['19dwz', '20aai']  # List of supernovae to fit
fix_z = True  # Fix redshift from redshifts.dat or targets.dat
fit_sigma = False  # Whether to fit an error scaling parameter

# Prior bounds
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

# Load data for multiple supernovae
print(f"Loading data for supernovae: {SN_NAMES}")
multi_sn_data = load_multiple_supernovae(SN_NAMES, fix_z=fix_z)

# Create SALT3 source for bandflux calculations (ONE instance before JIT)
source = SALT3Source()

n_sne = multi_sn_data['n_sne']
n_bands = multi_sn_data['n_bands']
bridges = multi_sn_data['bridges']
unique_bands = multi_sn_data['unique_bands']

print(f"Loaded {n_sne} supernovae with {n_bands} unique bands: {unique_bands}")

# Extract concatenated data arrays
all_times = multi_sn_data['all_times']
all_fluxes = multi_sn_data['all_fluxes']
all_fluxerrs = multi_sn_data['all_fluxerrs']
all_zps = multi_sn_data['all_zps']
all_band_indices = multi_sn_data['all_band_indices']
all_sn_indices = multi_sn_data['sn_indices']

# Individual SN data for easier access
times_list = multi_sn_data['times_list']
fluxes_list = multi_sn_data['fluxes_list']
fluxerrs_list = multi_sn_data['fluxerrs_list']
zps_list = multi_sn_data['zps_list']
band_indices_list = multi_sn_data['band_indices_list']
fixed_z_list = multi_sn_data['fixed_z_list']

# Define parameter structure
# For N SNe: params[0:4N] are SN parameters (t0, log_x0, x1, c for each SN)
# params[4N:4N+n_bands] are shared shift parameters
# params[4N+n_bands] is optional log_sigma

# Build parameter bounds and priors
param_bounds = {}
prior_dists = {}

# Add bounds for each SN's individual parameters
for i in range(n_sne):
    sn_name = multi_sn_data['sn_names'][i]
    if fix_z:
        param_bounds[f'{sn_name}_t0'] = (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max'])
        param_bounds[f'{sn_name}_x0'] = (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max'])
        param_bounds[f'{sn_name}_x1'] = (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max'])
        param_bounds[f'{sn_name}_c'] = (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
        
        prior_dists[f'{sn_name}_t0'] = distrax.Uniform(low=param_bounds[f'{sn_name}_t0'][0], 
                                                        high=param_bounds[f'{sn_name}_t0'][1])
        prior_dists[f'{sn_name}_x0'] = distrax.Uniform(low=param_bounds[f'{sn_name}_x0'][0], 
                                                        high=param_bounds[f'{sn_name}_x0'][1])
        prior_dists[f'{sn_name}_x1'] = distrax.Uniform(low=param_bounds[f'{sn_name}_x1'][0], 
                                                        high=param_bounds[f'{sn_name}_x1'][1])
        prior_dists[f'{sn_name}_c'] = distrax.Uniform(low=param_bounds[f'{sn_name}_c'][0], 
                                                       high=param_bounds[f'{sn_name}_c'][1])
    else:
        param_bounds[f'{sn_name}_z'] = (PRIOR_BOUNDS['z']['min'], PRIOR_BOUNDS['z']['max'])
        param_bounds[f'{sn_name}_t0'] = (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max'])
        param_bounds[f'{sn_name}_x0'] = (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max'])
        param_bounds[f'{sn_name}_x1'] = (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max'])
        param_bounds[f'{sn_name}_c'] = (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
        
        prior_dists[f'{sn_name}_z'] = distrax.Uniform(low=param_bounds[f'{sn_name}_z'][0], 
                                                       high=param_bounds[f'{sn_name}_z'][1])
        prior_dists[f'{sn_name}_t0'] = distrax.Uniform(low=param_bounds[f'{sn_name}_t0'][0], 
                                                        high=param_bounds[f'{sn_name}_t0'][1])
        prior_dists[f'{sn_name}_x0'] = distrax.Uniform(low=param_bounds[f'{sn_name}_x0'][0], 
                                                        high=param_bounds[f'{sn_name}_x0'][1])
        prior_dists[f'{sn_name}_x1'] = distrax.Uniform(low=param_bounds[f'{sn_name}_x1'][0], 
                                                        high=param_bounds[f'{sn_name}_x1'][1])
        prior_dists[f'{sn_name}_c'] = distrax.Uniform(low=param_bounds[f'{sn_name}_c'][0], 
                                                       high=param_bounds[f'{sn_name}_c'][1])

# Add shared shift parameters
for i, band in enumerate(unique_bands):
    param_bounds[f'shift_{band}'] = (PRIOR_BOUNDS['shift']['min'], PRIOR_BOUNDS['shift']['max'])
    prior_dists[f'shift_{band}'] = distrax.Uniform(low=param_bounds[f'shift_{band}'][0], 
                                                    high=param_bounds[f'shift_{band}'][1])

# Add optional sigma parameter
if fit_sigma:
    param_bounds['log_sigma'] = (PRIOR_BOUNDS['log_sigma']['min'], PRIOR_BOUNDS['log_sigma']['max'])
    prior_dists['log_sigma'] = distrax.Uniform(low=param_bounds['log_sigma'][0], 
                                               high=param_bounds['log_sigma'][1])

# Calculate total number of parameters
if fix_z:
    n_params_per_sn = 4  # t0, x0, x1, c
else:
    n_params_per_sn = 5  # z, t0, x0, x1, c

n_params_total = n_sne * n_params_per_sn + n_bands
if fit_sigma:
    n_params_total += 1

# NS settings - set n_live based on number of parameters
n_live = n_params_total * 25
NS_SETTINGS = {
    'n_live': n_live,
    'n_delete': n_live // 2,  # Set to n_live/2 for efficiency
    'num_mcmc_steps_multiplier': 5
}

print(f"Total parameters: {n_params_total}")
print(f"  - {n_sne} SNe × {n_params_per_sn} params/SN = {n_sne * n_params_per_sn}")
print(f"  - {n_bands} shared shift parameters")
if fit_sigma:
    print(f"  - 1 error scaling parameter")
print(f"\nNested sampling configuration:")
print(f"  - n_live: {NS_SETTINGS['n_live']} (25 × {n_params_total} parameters)")
print(f"  - n_delete: {NS_SETTINGS['n_delete']}")

@jax.jit
def logprior(params):
    """Calculate log prior probability for multi-SN parameters."""
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        return jax.vmap(logprior)(params)
    
    logp = 0.0
    idx = 0
    
    # Prior for each SN's parameters
    for sn_idx in range(n_sne):
        sn_name = multi_sn_data['sn_names'][sn_idx]
        
        if not fix_z:
            logp += prior_dists[f'{sn_name}_z'].log_prob(params[idx])
            idx += 1
        
        logp += prior_dists[f'{sn_name}_t0'].log_prob(params[idx])
        idx += 1
        logp += prior_dists[f'{sn_name}_x0'].log_prob(params[idx])
        idx += 1
        logp += prior_dists[f'{sn_name}_x1'].log_prob(params[idx])
        idx += 1
        logp += prior_dists[f'{sn_name}_c'].log_prob(params[idx])
        idx += 1
    
    # Prior for shared shift parameters
    for band in unique_bands:
        logp += prior_dists[f'shift_{band}'].log_prob(params[idx])
        idx += 1
    
    # Prior for optional sigma
    if fit_sigma:
        logp += prior_dists['log_sigma'].log_prob(params[idx])
    
    return logp

@jax.jit
def compute_multi_sn_loglikelihood(params):
    """Compute combined log likelihood for multiple SNe with shared shifts.

    Uses SALT3Source with v3.0 functional API for each supernova.
    """
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        return jax.vmap(compute_multi_sn_loglikelihood)(params)

    total_log_likelihood = 0.0
    idx = 0

    # Extract shared shift parameters (same for all SNe)
    shift_start_idx = n_sne * n_params_per_sn
    shifts = params[shift_start_idx:shift_start_idx + n_bands]

    # Extract optional sigma
    if fit_sigma:
        log_sigma = params[shift_start_idx + n_bands]
        sigma = 10 ** log_sigma
    else:
        sigma = 1.0

    # Calculate likelihood for each SN
    for sn_idx in range(n_sne):
        sn_name = multi_sn_data['sn_names'][sn_idx]

        # Extract this SN's parameters
        if fix_z:
            z = fixed_z_list[sn_idx][0]
            t0 = params[idx]
            idx += 1
            log_x0 = params[idx]
            idx += 1
            x1 = params[idx]
            idx += 1
            c = params[idx]
            idx += 1
        else:
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

        x0 = 10 ** log_x0

        # Create parameter dict for v3.0 functional API (x0, x1, c only)
        param_dict = {'x0': x0, 'x1': x1, 'c': c}

        # Get this SN's data
        times = times_list[sn_idx]
        fluxes = fluxes_list[sn_idx]
        fluxerrs = fluxerrs_list[sn_idx]
        zps = zps_list[sn_idx]
        band_indices = band_indices_list[sn_idx]

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
            shifts=shifts  # Add transmission shifts
        )

        # Calculate chi2 for this SN
        eff_fluxerrs = sigma * fluxerrs
        chi2 = jnp.sum(((fluxes - model_fluxes) / eff_fluxerrs) ** 2)
        log_likelihood_sn = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * eff_fluxerrs ** 2)))

        total_log_likelihood += log_likelihood_sn

    return total_log_likelihood

def sample_from_priors(rng_key, n_samples):
    """Sample from all prior distributions for multi-SN model."""
    samples = []
    
    n_keys = n_params_total
    keys = jax.random.split(rng_key, n_keys)
    
    idx = 0
    
    # Sample for each SN's parameters
    for sn_idx in range(n_sne):
        sn_name = multi_sn_data['sn_names'][sn_idx]
        
        if not fix_z:
            samples.append(prior_dists[f'{sn_name}_z'].sample(seed=keys[idx], sample_shape=(n_samples,)))
            idx += 1
        
        samples.append(prior_dists[f'{sn_name}_t0'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists[f'{sn_name}_x0'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists[f'{sn_name}_x1'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
        samples.append(prior_dists[f'{sn_name}_c'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
    
    # Sample shared shift parameters
    for band in unique_bands:
        samples.append(prior_dists[f'shift_{band}'].sample(seed=keys[idx], sample_shape=(n_samples,)))
        idx += 1
    
    # Sample optional sigma
    if fit_sigma:
        samples.append(prior_dists['log_sigma'].sample(seed=keys[idx], sample_shape=(n_samples,)))
    
    return jnp.column_stack(samples)

# Set up nested sampling
num_mcmc_steps = n_params_total * NS_SETTINGS['num_mcmc_steps_multiplier']

print("Setting up nested sampling algorithm...")
algo = blackjax.ns.adaptive.nss(
    logprior_fn=logprior,
    loglikelihood_fn=compute_multi_sn_loglikelihood,
    n_delete=NS_SETTINGS['n_delete'],
    num_mcmc_steps=num_mcmc_steps,
)

# Initialize random key and particles
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)

initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'])
print(f"Initial particles generated, shape: {initial_particles.shape}")

# Initialize state
state = algo.init(initial_particles, compute_multi_sn_loglikelihood)

# Define one_step function
@jax.jit
def one_step(carry, xs):
    """One step of nested sampling."""
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

# Process results
dead = jax.tree_map(lambda *args: jnp.concatenate(args), *dead)
logw = log_weights(rng_key, dead)
logZs = jax.scipy.special.logsumexp(logw, axis=0)

print(f"Runtime evidence: {state.sampler_state.logZ:.2f}")
print(f"Estimated evidence: {logZs.mean():.2f} ± {logZs.std():.2f}")

# Save chains
param_names = []

# Add SN parameter names
for sn_idx in range(n_sne):
    sn_name = multi_sn_data['sn_names'][sn_idx]
    if not fix_z:
        param_names.append(f'{sn_name}_z')
    param_names.extend([f'{sn_name}_t0', f'{sn_name}_log_x0', f'{sn_name}_x1', f'{sn_name}_c'])

# Add shift parameter names
for band in unique_bands:
    param_names.append(f'shift_{band}')

if fit_sigma:
    param_names.append('log_sigma')

save_chains_dead_birth(dead, param_names)

# Read chains and create visualizations
print("\nCreating corner plot...")
samples = read_chains('chains/chains', columns=param_names)

# Create corner plot
fig, axes = make_2d_axes(param_names, figsize=(16, 16), facecolor='w')
samples.plot_2d(axes, alpha=0.9, label="posterior")

# Add legend
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), 
                       loc='lower center', ncols=2)
plt.suptitle(f'Multi-SN SALT3 Parameters with Shared Transmission Shifts\n({", ".join(SN_NAMES)})', 
             y=1.02, fontsize=14)

# Save plot
plt.savefig('corner_plot_multi_sn_shifts.png', dpi=300, bbox_inches='tight')
plt.close()

# Print parameter statistics
print("\nParameter Statistics:")
print("-" * 80)
print(f"{'Parameter':<25} {'Mean':>15} {'Std Dev':>15}")
print("-" * 80)

# Print individual SN parameters
for sn_idx in range(n_sne):
    sn_name = multi_sn_data['sn_names'][sn_idx]
    print(f"\n{sn_name}:")
    sn_params = [p for p in param_names if p.startswith(sn_name)]
    for param in sn_params:
        mean = samples[param].mean()
        std = samples[param].std()
        param_display = param.replace(f'{sn_name}_', '  ')
        print(f"{param_display:<25} {mean:>15.6f} {std:>15.6f}")

# Print shared shift parameters
print("\nShared Transmission Shifts:")
for band in unique_bands:
    param = f'shift_{band}'
    mean = samples[param].mean()
    std = samples[param].std()
    print(f"  {band:<23} {mean:>15.6f} {std:>15.6f}")

if fit_sigma:
    mean = samples['log_sigma'].mean()
    std = samples['log_sigma'].std()
    print(f"\n{'log_sigma':<25} {mean:>15.6f} {std:>15.6f}")

print("-" * 80)

# Save statistics to file
with open('parameter_statistics_multi_sn_shifts.txt', 'w') as f:
    f.write(f"Multi-Supernova Parameter Statistics\n")
    f.write(f"Supernovae: {', '.join(SN_NAMES)}\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Parameter':<25} {'Mean':>15} {'Std Dev':>15}\n")
    f.write("-" * 80 + "\n")
    
    for sn_idx in range(n_sne):
        sn_name = multi_sn_data['sn_names'][sn_idx]
        f.write(f"\n{sn_name}:\n")
        sn_params = [p for p in param_names if p.startswith(sn_name)]
        for param in sn_params:
            mean = samples[param].mean()
            std = samples[param].std()
            param_display = param.replace(f'{sn_name}_', '  ')
            f.write(f"{param_display:<25} {mean:>15.6f} {std:>15.6f}\n")
    
    f.write("\nShared Transmission Shifts:\n")
    for band in unique_bands:
        param = f'shift_{band}'
        mean = samples[param].mean()
        std = samples[param].std()
        f.write(f"  {band:<23} {mean:>15.6f} {std:>15.6f}\n")
    
    if fit_sigma:
        mean = samples['log_sigma'].mean()
        std = samples['log_sigma'].std()
        f.write(f"\n{'log_sigma':<25} {mean:>15.6f} {std:>15.6f}\n")
    
    f.write("-" * 80 + "\n")

print("\nParameter statistics saved to 'parameter_statistics_multi_sn_shifts.txt'")
print("Corner plot saved to 'corner_plot_multi_sn_shifts.png'")