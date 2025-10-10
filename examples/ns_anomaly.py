"""
Anomaly Detection for Supernova Light Curves

Runs two nested sampling procedures:
1. Standard: fits SALT3 model parameters
2. Anomaly: includes log_p parameter to identify outliers

The anomaly model assigns each data point a probability of being an outlier,
allowing identification of problematic observations.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.data import load_and_process_data
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# Configuration
PRIOR_BOUNDS = {
    't0': (58000.0, 60000.0),
    'x0': (-5.0, -1.0),
    'x1': (-10.0, 10.0),
    'c': (-0.6, 0.6),
    'log_p': (-20.0, -1.0)
}

NS_SETTINGS = {
    'n_delete': 75,
    'n_live': 150,
    'num_mcmc_steps_multiplier': 5,
    'max_iterations': 500
}

SN_NAME = '19dwz'
DATA_DIR = 'data'

jax.config.update("jax_enable_x64", True)

# Load data with fixed redshift
times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = \
    load_and_process_data(SN_NAME, data_dir=DATA_DIR, fix_z=True)

z = fixed_z[0]

# Setup priors
base_params = ['t0', 'x0', 'x1', 'c']
anomaly_params = base_params + ['log_p']

standard_prior_dists = {p: distrax.Uniform(low=PRIOR_BOUNDS[p][0], high=PRIOR_BOUNDS[p][1])
                        for p in base_params}
anomaly_prior_dists = {p: distrax.Uniform(low=PRIOR_BOUNDS[p][0], high=PRIOR_BOUNDS[p][1])
                       for p in anomaly_params}


# Standard likelihood functions
@jax.jit
def logprior_standard(params):
    """Standard log prior (4 parameters)."""
    params = jnp.atleast_2d(params)
    logp = sum(standard_prior_dists[base_params[i]].log_prob(params[:, i])
               for i in range(len(base_params)))
    return jnp.reshape(logp, (-1,))


@jax.jit
def compute_single_loglikelihood_standard(params):
    """Standard Gaussian log likelihood."""
    t0, log_x0, x1, c = params
    x0 = 10 ** log_x0
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}

    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]

    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs) ** 2)
    return -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * fluxerrs ** 2)))


@jax.jit
def loglikelihood_standard(params):
    """Vectorized standard likelihood."""
    params = jnp.atleast_2d(params)
    return jax.vmap(compute_single_loglikelihood_standard)(params).reshape(-1,)


# Anomaly detection likelihood functions
@jax.jit
def logprior_anomaly(params):
    """Anomaly log prior (5 parameters including log_p)."""
    params = jnp.atleast_2d(params)
    logp = sum(anomaly_prior_dists[anomaly_params[i]].log_prob(params[:, i])
               for i in range(len(anomaly_params)))
    return jnp.reshape(logp, (-1,))


@jax.jit
def compute_single_loglikelihood_anomaly(params):
    """
    Anomaly detection likelihood with outlier probability.

    Returns tuple of (log_likelihood, emax) where emax indicates
    which points are likely normal (True) vs outliers (False).
    """
    t0, log_x0, x1, c, log_p = params
    x0 = 10 ** log_x0
    p = jnp.exp(log_p)
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}

    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]

    # Point-wise log likelihood with outlier probability
    point_logL = (-0.5 * ((fluxes - model_fluxes) / fluxerrs) ** 2
                  - 0.5 * jnp.log(2 * jnp.pi * fluxerrs ** 2) + jnp.log(1 - p))

    delta = jnp.max(jnp.abs(fluxes))
    emax = point_logL > (log_p - jnp.log(delta))
    logL = jnp.where(emax, point_logL, log_p - jnp.log(delta))

    return jnp.sum(logL), emax


@jax.jit
def loglikelihood_anomaly(params):
    """Vectorized anomaly likelihood."""
    params = jnp.atleast_2d(params)
    batch_loglike, _ = jax.vmap(compute_single_loglikelihood_anomaly)(params)
    return jnp.reshape(batch_loglike, (-1,))


def sample_from_priors(rng_key, n_samples, is_anomaly=False):
    """Sample from prior distributions."""
    prior_dists = anomaly_prior_dists if is_anomaly else standard_prior_dists
    param_list = anomaly_params if is_anomaly else base_params

    keys = jax.random.split(rng_key, len(param_list))
    samples = [prior_dists[param_list[i]].sample(seed=keys[i], sample_shape=(n_samples,))
               for i in range(len(param_list))]
    return jnp.column_stack(samples)


def run_nested_sampling(is_anomaly, output_prefix):
    """Run nested sampling for either standard or anomaly model."""
    print(f"\nRunning {output_prefix} nested sampling...")

    logprior_fn = logprior_anomaly if is_anomaly else logprior_standard
    loglikelihood_fn = loglikelihood_anomaly if is_anomaly else loglikelihood_standard
    param_names = anomaly_params if is_anomaly else base_params
    n_params = len(param_names)

    # Initialize algorithm
    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_inner_steps=n_params * NS_SETTINGS['num_mcmc_steps_multiplier'],
        num_delete=NS_SETTINGS['n_delete'],
    )

    # Initialize particles
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'], is_anomaly)
    state = algo.init(initial_particles)

    @jax.jit
    def one_step(carry, _):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    # Run sampling
    dead = []
    emax_values = [] if is_anomaly else None

    with tqdm.tqdm(desc=output_prefix, unit=" points",
                   total=NS_SETTINGS['max_iterations']*NS_SETTINGS['n_delete']) as pbar:
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

    # Process results
    dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
    logw = log_weights(rng_key, dead)
    logZs = jax.scipy.special.logsumexp(logw, axis=0)

    print(f"Evidence: {logZs.mean():.2f} ± {logZs.std():.2f}")

    # Save chains
    output_dir = f'results/chains_{SN_NAME}'
    os.makedirs(output_dir, exist_ok=True)

    points = np.array(dead.particles)
    logls_death = np.array(dead.logL)
    logls_birth = np.array(dead.logL_birth)
    data = np.column_stack([points, logls_death, logls_birth])

    chains_file = f'{output_dir}/{output_prefix}_dead-birth.txt'
    np.savetxt(chains_file, data)

    # Save weighted emax values for anomaly detection
    if is_anomaly and emax_values:
        emax_array = jnp.stack(emax_values)
        weights = jnp.exp(logw - jax.scipy.special.logsumexp(logw))

        if weights.ndim > 1:
            weights = weights[:, 0]

        min_len = min(len(emax_array), len(weights))
        emax_array = emax_array[:min_len]
        weights = weights[:min_len]

        weighted_emax = jnp.sum(emax_array * weights[:, None], axis=0) / jnp.sum(weights)
        emax_file = f'{output_dir}/{output_prefix}_weighted_emax.txt'
        np.savetxt(emax_file, weighted_emax)
        print(f"Saved weighted emax to {emax_file}")

    return param_names


# Run both versions
print(f"Analyzing {SN_NAME} with fixed z={z:.4f}")
anomaly_param_names = run_nested_sampling(is_anomaly=True, output_prefix='chains_anomaly')
standard_param_names = run_nested_sampling(is_anomaly=False, output_prefix='chains_standard')

# Create comparison plots
print("\nGenerating plots...")
output_dir = f'results/chains_{SN_NAME}'

# Load chains
try:
    standard_samples = read_chains(f'{output_dir}/chains_standard', columns=standard_param_names)
    anomaly_samples = read_chains(f'{output_dir}/chains_anomaly', columns=anomaly_param_names)

    # Corner plot comparison (base parameters only)
    fig, axes = make_2d_axes(base_params, figsize=(10, 10), facecolor='w')
    standard_samples.plot_2d(axes, alpha=0.7, label="Standard")
    anomaly_samples[base_params].plot_2d(axes, alpha=0.7, label="Anomaly")

    axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)),
                           loc='lower center', ncol=2)
    plt.suptitle(f'{SN_NAME} Parameter Comparison', fontsize=14)
    plt.savefig(f'{output_dir}/corner_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot weighted emax
    weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
    plt.figure(figsize=(12, 4))
    plt.plot(weighted_emax, 'k-', linewidth=2)
    plt.fill_between(range(len(weighted_emax)), 0, weighted_emax, alpha=0.3)
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Threshold')
    plt.xlabel('Data Point Index')
    plt.ylabel('Weighted Emax (outlier probability)')
    plt.title(f'{SN_NAME} Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/weighted_emax.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}/")
    print(f"Points below threshold: {np.sum(weighted_emax < 0.2)}/{len(weighted_emax)}")

except Exception as e:
    print(f"Warning: Could not create plots - {e}")

print("\nAnomaly detection complete!")
