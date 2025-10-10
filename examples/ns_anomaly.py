"""
Anomaly detection for supernova light curves using nested sampling.

This script implements a Bayesian anomaly detection framework that:
1. Runs standard SALT3 nested sampling
2. Runs anomaly detection nested sampling with an additional log_p parameter
3. Identifies potential outlier data points using weighted emax values
4. Compares the two approaches via corner plots and light curve fits

The anomaly detection adds a parameter log_p that allows individual data points
to be down-weighted if they don't fit the model well.
"""
import distrax
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
from blackjax.ns.utils import log_weights
from jax_supernovae import SALT3Source
from jax_supernovae.data import load_and_process_data
from jax_supernovae.utils import save_chains_dead_birth
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# ============================================================================
# Configuration
# ============================================================================

# Supernova to analyze
SN_NAME = '19dwz'

# Nested sampling settings
NS_SETTINGS = {
    'n_delete': 60,
    'n_live': 125,
    'num_mcmc_steps_multiplier': 5,
    'max_iterations': 500
}

# Prior bounds
PRIOR_BOUNDS = {
    't0': {'min': 58000.0, 'max': 59000.0},
    'x0': {'min': -5.0, 'max': -2.6},
    'x1': {'min': -4.0, 'max': 4.0},
    'c': {'min': -0.3, 'max': 0.3},
    'log_p': {'min': -20, 'max': -1}
}

# Whether to fix redshift (True = use spectroscopic z)
FIX_Z = True

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# ============================================================================
# Load data
# ============================================================================

print(f"Loading data for {SN_NAME}...")
times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = \
    load_and_process_data(SN_NAME, data_dir='data', fix_z=FIX_Z)

print(f"Loaded {len(times)} data points across {len(unique_bands)} bands: {unique_bands}")
if FIX_Z:
    print(f"Using fixed redshift: z = {fixed_z[0]:.4f}")

# Create SALT3 source
source = SALT3Source()

# ============================================================================
# Set up priors
# ============================================================================

# Standard model (no anomaly detection)
standard_prior_dists = {
    't0': distrax.Uniform(low=PRIOR_BOUNDS['t0']['min'], high=PRIOR_BOUNDS['t0']['max']),
    'x0': distrax.Uniform(low=PRIOR_BOUNDS['x0']['min'], high=PRIOR_BOUNDS['x0']['max']),
    'x1': distrax.Uniform(low=PRIOR_BOUNDS['x1']['min'], high=PRIOR_BOUNDS['x1']['max']),
    'c': distrax.Uniform(low=PRIOR_BOUNDS['c']['min'], high=PRIOR_BOUNDS['c']['max'])
}

# Anomaly model (includes log_p parameter)
anomaly_prior_dists = {
    **standard_prior_dists,
    'log_p': distrax.Uniform(low=PRIOR_BOUNDS['log_p']['min'], high=PRIOR_BOUNDS['log_p']['max'])
}

# ============================================================================
# Likelihood functions
# ============================================================================

@jax.jit
def logprior_standard(params):
    """Calculate log prior for standard model."""
    # Handle both single and batched inputs
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        return jax.vmap(logprior_standard)(params)

    logp = (standard_prior_dists['t0'].log_prob(params[0]) +
            standard_prior_dists['x0'].log_prob(params[1]) +
            standard_prior_dists['x1'].log_prob(params[2]) +
            standard_prior_dists['c'].log_prob(params[3]))
    return logp

@jax.jit
def logprior_anomaly(params):
    """Calculate log prior for anomaly detection model."""
    # Handle both single and batched inputs
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        return jax.vmap(logprior_anomaly)(params)

    logp = (anomaly_prior_dists['t0'].log_prob(params[0]) +
            anomaly_prior_dists['x0'].log_prob(params[1]) +
            anomaly_prior_dists['x1'].log_prob(params[2]) +
            anomaly_prior_dists['c'].log_prob(params[3]) +
            anomaly_prior_dists['log_p'].log_prob(params[4]))
    return logp

@jax.jit
def compute_single_loglikelihood_standard(params):
    """Compute log likelihood for standard model."""
    # Handle both single and batched inputs
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        return jax.vmap(compute_single_loglikelihood_standard)(params)

    t0, log_x0, x1, c = params
    z = fixed_z[0]
    x0 = 10 ** log_x0

    # Calculate model fluxes
    param_dict = {'x0': x0, 'x1': x1, 'c': c}
    phases = (times - t0) / (1 + z)

    model_fluxes = source.bandflux(
        param_dict, None, phases, zp=zps, zpsys='ab',
        band_indices=band_indices, bridges=bridges, unique_bands=unique_bands
    )

    # Gaussian likelihood
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs) ** 2)
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * fluxerrs ** 2)))
    return log_likelihood

@jax.jit
def compute_single_loglikelihood_anomaly(params):
    """Compute log likelihood with anomaly detection.

    Returns
    -------
    log_likelihood : float
        Total log likelihood
    emax : array
        Boolean array indicating which data points are considered normal
    """
    # Handle both single and batched inputs
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        # For batched inputs, vmap over the function
        batch_loglike, batch_emax = jax.vmap(compute_single_loglikelihood_anomaly)(params)
        return batch_loglike, batch_emax

    t0, log_x0, x1, c, log_p = params
    z = fixed_z[0]
    x0 = 10 ** log_x0
    p = jnp.exp(log_p)

    # Calculate model fluxes
    param_dict = {'x0': x0, 'x1': x1, 'c': c}
    phases = (times - t0) / (1 + z)

    model_fluxes = source.bandflux(
        param_dict, None, phases, zp=zps, zpsys='ab',
        band_indices=band_indices, bridges=bridges, unique_bands=unique_bands
    )

    # Per-point likelihood with anomaly model
    point_logL = (-0.5 * ((fluxes - model_fluxes) / fluxerrs) ** 2
                  - 0.5 * jnp.log(2 * jnp.pi * fluxerrs ** 2)
                  + jnp.log(1 - p))

    # Identify which points are well-fit (emax = True means normal)
    delta = jnp.max(jnp.abs(fluxes))
    emax = point_logL > (log_p - jnp.log(delta))

    # Total likelihood
    logL = jnp.where(emax, point_logL, log_p - jnp.log(delta))
    return jnp.sum(logL), emax

# ============================================================================
# Sampling from priors
# ============================================================================

def sample_from_priors(rng_key, n_samples, is_anomaly=False):
    """Sample from prior distributions."""
    if is_anomaly:
        keys = jax.random.split(rng_key, 5)
        return jnp.column_stack([
            anomaly_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
            anomaly_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
            anomaly_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
            anomaly_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
            anomaly_prior_dists['log_p'].sample(seed=keys[4], sample_shape=(n_samples,))
        ])
    else:
        keys = jax.random.split(rng_key, 4)
        return jnp.column_stack([
            standard_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
            standard_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
            standard_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
            standard_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,))
        ])

# ============================================================================
# Run nested sampling
# ============================================================================

def run_nested_sampling(logprior_fn, loglikelihood_fn, is_anomaly=False, label=""):
    """Run nested sampling and save results."""

    print(f"\n{'='*60}")
    print(f"Running {label} nested sampling...")
    print(f"{'='*60}")

    n_params = 5 if is_anomaly else 4
    num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']

    # Initialize algorithm
    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_inner_steps=num_mcmc_steps,
        num_delete=NS_SETTINGS['n_delete'],
    )

    # Initialize particles
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'], is_anomaly)

    print(f"Initial particles shape: {initial_particles.shape}")

    # Initialize state
    state = algo.init(initial_particles)

    # Define one step
    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    # Run sampling loop
    dead = []
    emax_values = [] if is_anomaly else None

    print("Running sampling...")
    with tqdm.tqdm(desc="Dead points", unit=" pts") as pbar:
        for i in range(NS_SETTINGS['max_iterations']):
            if state.logZ_live - state.logZ < -3:
                break

            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(NS_SETTINGS['n_delete'])

            # Store emax values for anomaly detection
            if is_anomaly:
                for j in range(len(dead_info.particles)):
                    _, emax = compute_single_loglikelihood_anomaly(dead_info.particles[j])
                    emax_values.append(emax)

            if i % 10 == 0:
                print(f"Iteration {i}: logZ = {state.logZ:.2f}")

    # Process results
    dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
    logw = log_weights(rng_key, dead)
    logZs = jax.scipy.special.logsumexp(logw, axis=0)

    print(f"\nRuntime evidence: {state.logZ:.2f}")
    print(f"Estimated evidence: {logZs.mean():.2f} ± {logZs.std():.2f}")

    # Save chains
    output_dir = f'chains_{SN_NAME}'
    os.makedirs(output_dir, exist_ok=True)

    param_names = ['t0', 'log_x0', 'x1', 'c']
    if is_anomaly:
        param_names.append('log_p')

    # Save dead-birth chains using utility function
    # Change to output_dir, save chains with label as root_dir, then change back
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    save_chains_dead_birth(dead, param_names, root_dir=label)
    chain_dir = label  # For use in emax saving below
    os.chdir(original_cwd)

    # Save weighted emax for anomaly detection
    if is_anomaly and emax_values:
        emax_array = jnp.stack(emax_values)
        weights = jnp.exp(logw - jax.scipy.special.logsumexp(logw))

        if weights.ndim > 1:
            weights = weights[:, 0]

        # Ensure compatible shapes
        min_len = min(len(emax_array), len(weights))
        emax_array = emax_array[:min_len]
        weights = weights[:min_len]

        # Calculate weighted average
        weighted_emax = jnp.zeros(emax_array.shape[1])
        for i in range(emax_array.shape[1]):
            weighted_emax = weighted_emax.at[i].set(
                jnp.sum(emax_array[:, i] * weights) / jnp.sum(weights)
            )

        emax_file = os.path.join(output_dir, label, f'{label}_weighted_emax.txt')
        np.savetxt(emax_file, weighted_emax)
        print(f"Saved weighted emax to {emax_file}")

    return param_names, output_dir

# Wrapper for anomaly likelihood that returns only logL (not emax)
@jax.jit
def loglikelihood_anomaly(params):
    """Wrapper for anomaly likelihood that discards emax."""
    logL, _ = compute_single_loglikelihood_anomaly(params)
    return logL

# Run both versions
standard_params, output_dir = run_nested_sampling(
    logprior_standard, compute_single_loglikelihood_standard,
    is_anomaly=False, label="standard"
)

anomaly_params, _ = run_nested_sampling(
    logprior_anomaly, loglikelihood_anomaly,
    is_anomaly=True, label="anomaly"
)

# ============================================================================
# Create visualizations
# ============================================================================

print(f"\n{'='*60}")
print("Creating visualizations...")
print(f"{'='*60}")

# Load chains (chains are saved as {root_dir}/{root_dir}_dead-birth.txt)
standard_samples = read_chains(f'{output_dir}/standard/standard', columns=standard_params)
anomaly_samples = read_chains(f'{output_dir}/anomaly/anomaly', columns=anomaly_params)

# 1. Corner plot comparison (excluding log_p)
print("\nCreating corner plot comparison...")
fig, axes = make_2d_axes(standard_params, figsize=(10, 10), facecolor='w')
standard_samples.plot_2d(axes, alpha=0.7, label="Standard")
anomaly_samples[standard_params].plot_2d(axes, alpha=0.7, label="Anomaly")
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)),
                        loc='lower center', ncol=2)
plt.suptitle('SALT3 Parameter Posteriors: Standard vs Anomaly', y=1.02, fontsize=14)
plt.savefig(f'{output_dir}/corner_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Anomaly-specific corner plot (including log_p)
print("Creating anomaly corner plot with log_p...")
fig, axes = make_2d_axes(anomaly_params, figsize=(12, 12), facecolor='w')
anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
plt.suptitle('Anomaly Detection Parameters (including log_p)', y=1.02, fontsize=14)
plt.savefig(f'{output_dir}/corner_anomaly_logp.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Light curve with anomaly indicators
print("Creating light curve plot with anomaly detection...")
try:
    weighted_emax = np.loadtxt(f'{output_dir}/anomaly/anomaly_weighted_emax.txt')

    # Get median parameters from anomaly fit
    median_params = {
        't0': float(np.median(anomaly_samples['t0'])),
        'x0': 10 ** float(np.median(anomaly_samples['log_x0'])),
        'x1': float(np.median(anomaly_samples['x1'])),
        'c': float(np.median(anomaly_samples['c']))
    }

    # Create time grid for model curve
    t_grid = np.linspace(np.min(times) - 10, np.max(times) + 10, 200)
    phases_grid = (t_grid - median_params['t0']) / (1 + fixed_z[0])

    # Calculate model on grid
    param_dict = {'x0': median_params['x0'], 'x1': median_params['x1'], 'c': median_params['c']}

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Define colors for bands
    colors = ['g', 'orange', 'r', 'brown']
    markers = ['o', 's', 'D', '^']

    # Identify anomalous points (threshold = 0.2)
    threshold = 0.2
    all_times_sorted = np.sort(np.unique(times))

    # Plot each band
    for i, band_name in enumerate(unique_bands):
        mask = band_indices == i
        band_times = times[mask]
        band_fluxes = fluxes[mask]
        band_errors = fluxerrs[mask]

        # Map times to emax values
        time_indices = np.searchsorted(all_times_sorted, band_times)
        time_indices = np.clip(time_indices, 0, len(weighted_emax) - 1)
        point_emax = weighted_emax[time_indices]

        # Separate normal and anomalous points
        normal_mask = point_emax >= threshold
        anomaly_mask = point_emax < threshold

        # Plot normal points
        if np.any(normal_mask):
            ax1.errorbar(band_times[normal_mask], band_fluxes[normal_mask],
                        yerr=band_errors[normal_mask], fmt=markers[i], color=colors[i],
                        label=f'{band_name}', markersize=6, alpha=0.6)

        # Plot anomalous points as stars
        if np.any(anomaly_mask):
            ax1.errorbar(band_times[anomaly_mask], band_fluxes[anomaly_mask],
                        yerr=band_errors[anomaly_mask], fmt='*', color=colors[i],
                        markersize=12, alpha=0.8)

        # Calculate and plot model curve for this band
        model_fluxes_grid = source.bandflux(
            param_dict, band_name, phases_grid, zp=zps[0], zpsys='ab'
        )
        ax1.plot(t_grid, model_fluxes_grid, '-', color=colors[i], linewidth=2, alpha=0.7)

    ax1.set_ylabel('Flux', fontsize=12)
    ax1.set_title(f'Light Curve with Anomaly Detection (z = {fixed_z[0]:.4f})', fontsize=14)
    ax1.legend(ncol=2, fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot weighted emax
    ax2.plot(all_times_sorted, weighted_emax, 'k-', linewidth=2)
    ax2.fill_between(all_times_sorted, 0, weighted_emax, alpha=0.3)
    ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5,
                label=f'Threshold = {threshold}')
    ax2.set_xlabel('MJD', fontsize=12)
    ax2.set_ylabel('Weighted Emax', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Match x-axis limits
    ax1.set_xlim(ax2.get_xlim())

    plt.tight_layout()
    plt.savefig(f'{output_dir}/light_curve_anomaly.png', dpi=300, bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Warning: Could not create light curve plot - {str(e)}")

# Print summary statistics
print(f"\n{'='*60}")
print("Parameter Statistics")
print(f"{'='*60}")
print(f"{'Parameter':<12} {'Standard':>20} {'Anomaly':>20}")
print("-" * 60)
for param in standard_params:
    std_mean = standard_samples[param].mean()
    std_std = standard_samples[param].std()
    anom_mean = anomaly_samples[param].mean()
    anom_std = anomaly_samples[param].std()
    print(f"{param:<12} {std_mean:>10.4f} ± {std_std:<8.4f} {anom_mean:>10.4f} ± {anom_std:<8.4f}")

if 'log_p' in anomaly_samples.columns:
    logp_mean = anomaly_samples['log_p'].mean()
    logp_std = anomaly_samples['log_p'].std()
    print(f"{'log_p':<12} {'N/A':>20} {logp_mean:>10.4f} ± {logp_std:<8.4f}")

print(f"\nResults saved to {output_dir}/")
print("Generated plots:")
print(f"  - corner_comparison.png")
print(f"  - corner_anomaly_logp.png")
print(f"  - light_curve_anomaly.png")
