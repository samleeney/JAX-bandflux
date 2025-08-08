"""
Multi-supernova nested sampling for transmission shift calibration (Stage 1).

This script implements Stage 1 of the nuisance-free supernova calibration,
jointly fitting transmission shifts for filter bands across multiple SNe.
"""

import distrax
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import tqdm
import blackjax
import os
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.multi_sn_utils import (
    load_and_process_multiple_sne, 
    unpack_parameters,
    pack_parameters
)
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Configuration
SN_NAMES = ['19agl', '19ahi', '19ahr']  # Start with 3 SNe for testing
FIX_Z = True  # Fix redshift for all SNe

# Nested sampling settings
NS_SETTINGS = {
    'n_delete': 1,
    'n_live': 200,  # Increased for more parameters
    'num_mcmc_steps_multiplier': 5
}

# Prior bounds for transmission shifts (in Angstroms)
T_SHIFT_BOUNDS = {'min': -50.0, 'max': 50.0}

# Prior bounds for SN parameters (same as single SN case)
SN_PARAM_BOUNDS = {
    't0': {'min': 58000.0, 'max': 59000.0},
    'x0': {'min': -5.0, 'max': -2.6},
    'x1': {'min': -4.0, 'max': 4.0},
    'c': {'min': -0.3, 'max': 0.3},
}

print(f"Loading data for {len(SN_NAMES)} supernovae...")
sne_data, global_band_names, global_bridges = load_and_process_multiple_sne(
    SN_NAMES, data_dir='data', fix_z=FIX_Z
)

# Define dimensions
n_bands_global = len(global_band_names)
n_sne = len(SN_NAMES)
n_params_sn = 4  # t0, log_x0, x1, c per SN
n_params_total = n_bands_global + n_sne * n_params_sn

print(f"Global bands: {global_band_names}")
print(f"Total parameters: {n_params_total}")
print(f"  - Transmission shifts: {n_bands_global}")
print(f"  - SN parameters: {n_sne * n_params_sn} ({n_sne} SNe x {n_params_sn} params)")

# Create prior distributions
prior_dists = []

# Transmission shift priors (uniform)
for band_name in global_band_names:
    prior_dists.append(
        distrax.Uniform(low=T_SHIFT_BOUNDS['min'], high=T_SHIFT_BOUNDS['max'])
    )

# SN parameter priors (uniform for each SN)
for i in range(n_sne):
    prior_dists.append(distrax.Uniform(low=SN_PARAM_BOUNDS['t0']['min'], 
                                      high=SN_PARAM_BOUNDS['t0']['max']))
    prior_dists.append(distrax.Uniform(low=SN_PARAM_BOUNDS['x0']['min'], 
                                      high=SN_PARAM_BOUNDS['x0']['max']))
    prior_dists.append(distrax.Uniform(low=SN_PARAM_BOUNDS['x1']['min'], 
                                      high=SN_PARAM_BOUNDS['x1']['max']))
    prior_dists.append(distrax.Uniform(low=SN_PARAM_BOUNDS['c']['min'], 
                                      high=SN_PARAM_BOUNDS['c']['max']))

@jax.jit
def logprior(params):
    """Calculate log prior probability for all parameters."""
    logp = 0.0
    for i, dist in enumerate(prior_dists):
        logp += dist.log_prob(params[i])
    return logp

@jax.jit
def compute_joint_loglikelihood(params):
    """
    Computes the joint log-likelihood for multiple supernovae.
    
    Parameters
    ----------
    params : jnp.array
        Flat vector of all parameters (shifts + all SN params).
    
    Returns
    -------
    float
        Total log-likelihood.
    """
    # Unpack shared transmission shift parameters
    global_shifts = params[:n_bands_global]
    
    # Define the loop body function for a single supernova
    def sn_loop_body(i, total_log_l):
        # Get data for this SN
        times = sne_data['times'][i]
        fluxes = sne_data['fluxes'][i]
        fluxerrs = sne_data['fluxerrs'][i]
        zps = sne_data['zps'][i]
        band_indices = sne_data['band_indices'][i]
        valid_mask = sne_data['valid_mask'][i]
        z = sne_data['fixed_z'][i]
        local_to_global = sne_data['local_to_global_map'][i]
        n_bands_sn = sne_data['n_bands'][i].astype(jnp.int32)
        
        # Unpack this SN's SALT parameters
        start_idx = n_bands_global + i * n_params_sn
        t0 = params[start_idx]
        log_x0 = params[start_idx + 1]
        x1 = params[start_idx + 2]
        c = params[start_idx + 3]
        x0 = 10**log_x0
        
        param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
        
        # Gather the required shifts for this SN's bands
        # Since n_bands_sn is traced, we can't use it for dynamic slicing
        # Instead, get all shifts and rely on the model to use only what's needed
        shifts_for_sn = global_shifts[local_to_global]
        
        # Since bridges can't be dynamically indexed in JAX, we use the global bridges
        # and select based on local_to_global mapping
        # This means we calculate fluxes for all possible bands but only use the valid ones
        bridges_to_use = [global_bridges[idx] for idx in range(len(global_bridges))]
        
        # Calculate model fluxes for all global bands
        model_fluxes_all_global = optimized_salt3_multiband_flux(
            times,
            bridges=bridges_to_use,
            params=param_dict,
            zps=jnp.full(len(global_bridges), 23.9),  # Use default ZP for all
            zpsys='ab',
            shifts=global_shifts  # All global shifts
        )
        
        # Now select only the bands this SN actually uses
        # Map from local band indices to global band indices
        model_fluxes_all_bands = model_fluxes_all_global[:, local_to_global]
        
        # Select the correct band for each observation
        # This works correctly even with 1 band (all band_indices will be 0)
        row_indices = jnp.arange(len(times))
        # Ensure band_indices are within bounds for safety
        safe_band_indices = jnp.clip(band_indices, 0, jnp.maximum(n_bands_sn - 1, 0))
        model_fluxes = model_fluxes_all_bands[row_indices, safe_band_indices]
        
        # Calculate chi2 only for valid data points
        residuals = (fluxes - model_fluxes) / fluxerrs
        chi2_terms = jnp.where(valid_mask, residuals**2, 0.0)
        chi2 = jnp.sum(chi2_terms)
        
        # Calculate normalization term
        log_norm_terms = jnp.where(
            valid_mask, 
            jnp.log(2 * jnp.pi * fluxerrs**2), 
            0.0
        )
        log_norm = jnp.sum(log_norm_terms)
        
        log_l_sn = -0.5 * (chi2 + log_norm)
        
        return total_log_l + log_l_sn
    
    # Run the loop over all supernovae
    initial_log_l = 0.0
    total_log_l = lax.fori_loop(0, n_sne, sn_loop_body, initial_log_l)
    
    return total_log_l

def sample_from_priors(rng_key, n_samples):
    """Sample from all prior distributions."""
    keys = jax.random.split(rng_key, n_params_total)
    samples = []
    
    for i, dist in enumerate(prior_dists):
        samples.append(dist.sample(seed=keys[i], sample_shape=(n_samples,)))
    
    return jnp.column_stack(samples)

# Calculate MCMC steps based on total parameters
num_mcmc_steps = n_params_total * NS_SETTINGS['num_mcmc_steps_multiplier']

# Initialize nested sampling algorithm
print("\nSetting up nested sampling algorithm...")
print(f"  MCMC steps per iteration: {num_mcmc_steps}")

algo = blackjax.ns.adaptive.nss(
    logprior_fn=logprior,
    loglikelihood_fn=compute_joint_loglikelihood,
    n_delete=NS_SETTINGS['n_delete'],
    num_mcmc_steps=num_mcmc_steps,
)

# Initialize random key and particles
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)

print(f"Sampling {NS_SETTINGS['n_live']} initial particles...")
initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'])
print(f"Initial particles shape: {initial_particles.shape}")

# Initialize state
print("Initializing sampler state...")
state = algo.init(initial_particles, compute_joint_loglikelihood)

# Define one_step function with JIT
@jax.jit
def one_step(carry, xs):
    """One step of the nested sampling algorithm."""
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

# Run nested sampling
dead = []
print("\nRunning nested sampling...")
print("(This may take several minutes due to the increased parameter space)")

with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    max_iterations = 5000  # Safety limit
    iteration = 0
    
    while (not state.sampler_state.logZ_live - state.sampler_state.logZ < -3) and iteration < max_iterations:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(NS_SETTINGS['n_delete'])
        iteration += 1
        
        # Print progress periodically
        if iteration % 100 == 0:
            print(f"\n  Iteration {iteration}: logZ = {state.sampler_state.logZ:.2f}")

# Process results
print("\nProcessing results...")
dead = jax.tree_map(lambda *args: jnp.concatenate(args), *dead)
logw = log_weights(rng_key, dead)
logZs = jax.scipy.special.logsumexp(logw, axis=0)

print(f"\nRuntime evidence: {state.sampler_state.logZ:.2f}")
print(f"Estimated evidence: {logZs.mean():.2f} ± {logZs.std():.2f}")

# Build parameter names for saving
param_names = []
for band_name in global_band_names:
    param_names.append(f'T_{band_name}')

for i in range(n_sne):
    param_names.extend([
        f't0_{SN_NAMES[i]}',
        f'log_x0_{SN_NAMES[i]}',
        f'x1_{SN_NAMES[i]}',
        f'c_{SN_NAMES[i]}'
    ])

# Save chains
print(f"\nSaving chains with {len(param_names)} parameters...")
save_chains_dead_birth(dead, param_names)

# Read chains and create visualizations
print("\nCreating visualization...")
samples = read_chains('chains/chains', columns=param_names)

# Print transmission shift statistics
print("\n" + "="*60)
print("TRANSMISSION SHIFT RESULTS")
print("="*60)
print(f"{'Band':<10} {'Mean (Å)':>12} {'Std (Å)':>12} {'68% CI':>25}")
print("-"*60)

for band_name in global_band_names:
    param_name = f'T_{band_name}'
    mean = samples[param_name].mean()
    std = samples[param_name].std()
    percentiles = np.percentile(samples[param_name], [16, 84])
    ci_str = f"[{percentiles[0]:6.2f}, {percentiles[1]:6.2f}]"
    print(f"{band_name:<10} {mean:>12.2f} {std:>12.2f} {ci_str:>25}")

# Create a focused plot for transmission shifts only
n_shift_params = len(global_band_names)
shift_param_names = [f'T_{band}' for band in global_band_names]

if n_shift_params > 1:
    fig, axes = make_2d_axes(shift_param_names, figsize=(10, 10), facecolor='w')
    samples.plot_2d(axes, alpha=0.9, label="posterior", columns=shift_param_names)
    axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), 
                           loc='lower center', ncols=2)
    plt.suptitle('Transmission Shift Posterior Distributions', y=1.02, fontsize=14)
    plt.savefig('transmission_shifts_corner.png', dpi=300, bbox_inches='tight')
    print("\nTransmission shift corner plot saved to 'transmission_shifts_corner.png'")
    plt.close()

# Save summary statistics
with open('transmission_shift_results.txt', 'w') as f:
    f.write("Transmission Shift Calibration Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Supernovae analyzed: {', '.join(SN_NAMES)}\n")
    f.write(f"Global bands: {', '.join(global_band_names)}\n")
    f.write(f"Evidence: {logZs.mean():.2f} ± {logZs.std():.2f}\n\n")
    
    f.write(f"{'Band':<10} {'Mean (Å)':>12} {'Std (Å)':>12} {'68% CI':>25}\n")
    f.write("-" * 60 + "\n")
    
    for band_name in global_band_names:
        param_name = f'T_{band_name}'
        mean = samples[param_name].mean()
        std = samples[param_name].std()
        percentiles = np.percentile(samples[param_name], [16, 84])
        ci_str = f"[{percentiles[0]:6.2f}, {percentiles[1]:6.2f}]"
        f.write(f"{band_name:<10} {mean:>12.2f} {std:>12.2f} {ci_str:>25}\n")

print("Results saved to 'transmission_shift_results.txt'")
print("\nStage 1 calibration complete!")