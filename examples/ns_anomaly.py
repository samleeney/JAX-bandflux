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

# Try to load settings.yaml; if not found, use defaults.
try:
    with open('settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
except FileNotFoundError:
    settings = {}

# Set fix_z from settings (default True) and nested sampling settings.
fix_z = settings.get('fix_z', True)
if 'nested_sampling' in settings:
    NS_SETTINGS = settings['nested_sampling']
else:
    NS_SETTINGS = {
        'max_iterations': int(os.environ.get('NS_MAX_ITERATIONS', '2000')),
        'n_delete': 1,
        'n_live': 125,
        'num_mcmc_steps_multiplier': 5
    }

# Define default prior bounds; these may be overridden by settings.
PRIOR_BOUNDS = {
    'z': {'min': 0.001, 'max': 0.2},
    't0': {'min': 58000.0, 'max': 59000.0},
    'x0': {'min': -5.0, 'max': -2.6},
    'x1': {'min': -4.0, 'max': 4.0},
    'c': {'min': -0.3, 'max': 0.3},
    'sigma': {'min': 0.5, 'max': 1.5},
    'log_p': {'min': -3.0, 'max': 1.0}
}
if 'prior_bounds' in settings:
    PRIOR_BOUNDS = settings['prior_bounds']

# Option flag: when fit_sigma is True, an extra parameter is added.
fit_sigma = True

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load and process data
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz', data_dir='data', fix_z=fix_z)

# =============================================================================
# Set up parameter bounds and prior distributions for the standard (non‐anomaly)
# nested sampling version.
# =============================================================================
if fix_z:
    standard_param_bounds = {
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max']),
        'sigma': (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])
    }
else:
    standard_param_bounds = {
        'z': (PRIOR_BOUNDS['z']['min'], PRIOR_BOUNDS['z']['max']),
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max']),
        'sigma': (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])
    }
if fix_z:
    standard_prior_dists = {
        't0': distrax.Uniform(low=standard_param_bounds['t0'][0], high=standard_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=standard_param_bounds['x0'][0], high=standard_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=standard_param_bounds['x1'][0], high=standard_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=standard_param_bounds['c'][0], high=standard_param_bounds['c'][1]),
        'sigma': distrax.Uniform(low=standard_param_bounds['sigma'][0], high=standard_param_bounds['sigma'][1])
    }
else:
    standard_prior_dists = {
        'z': distrax.Uniform(low=standard_param_bounds['z'][0], high=standard_param_bounds['z'][1]),
        't0': distrax.Uniform(low=standard_param_bounds['t0'][0], high=standard_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=standard_param_bounds['x0'][0], high=standard_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=standard_param_bounds['x1'][0], high=standard_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=standard_param_bounds['c'][0], high=standard_param_bounds['c'][1]),
        'sigma': distrax.Uniform(low=standard_param_bounds['sigma'][0], high=standard_param_bounds['sigma'][1])
    }

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
        'sigma': (standard_param_bounds['sigma'][0], standard_param_bounds['sigma'][1]),
        'log_p': (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    }
else:
    anomaly_param_bounds = {
        'z': (standard_param_bounds['z'][0], standard_param_bounds['z'][1]),
        't0': (standard_param_bounds['t0'][0], standard_param_bounds['t0'][1]),
        'x0': (standard_param_bounds['x0'][0], standard_param_bounds['x0'][1]),
        'x1': (standard_param_bounds['x1'][0], standard_param_bounds['x1'][1]),
        'c': (standard_param_bounds['c'][0], standard_param_bounds['c'][1]),
        'sigma': (standard_param_bounds['sigma'][0], standard_param_bounds['sigma'][1]),
        'log_p': (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    }
if fix_z:
    anomaly_prior_dists = {
        't0': distrax.Uniform(low=anomaly_param_bounds['t0'][0], high=anomaly_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=anomaly_param_bounds['x0'][0], high=anomaly_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=anomaly_param_bounds['x1'][0], high=anomaly_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=anomaly_param_bounds['c'][0], high=anomaly_param_bounds['c'][1]),
        'sigma': distrax.Uniform(low=anomaly_param_bounds['sigma'][0], high=anomaly_param_bounds['sigma'][1]),
        'log_p': distrax.Uniform(low=anomaly_param_bounds['log_p'][0], high=anomaly_param_bounds['log_p'][1])
    }
else:
    anomaly_prior_dists = {
        'z': distrax.Uniform(low=anomaly_param_bounds['z'][0], high=anomaly_param_bounds['z'][1]),
        't0': distrax.Uniform(low=anomaly_param_bounds['t0'][0], high=anomaly_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=anomaly_param_bounds['x0'][0], high=anomaly_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=anomaly_param_bounds['x1'][0], high=anomaly_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=anomaly_param_bounds['c'][0], high=anomaly_param_bounds['c'][1]),
        'sigma': distrax.Uniform(low=anomaly_param_bounds['sigma'][0], high=anomaly_param_bounds['sigma'][1]),
        'log_p': distrax.Uniform(low=anomaly_param_bounds['log_p'][0], high=anomaly_param_bounds['log_p'][1])
    }

# =============================================================================
# Standard likelihood functions (using salt3 multiband flux).
# =============================================================================
@jax.jit
def logprior_standard(params):
    """Calculate log prior probability for standard nested sampling."""
    params = jnp.atleast_2d(params)
    if fix_z:
        logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 0])
        logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 1])
        logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 2])
        logp_c  = standard_prior_dists['c'].log_prob(params[:, 3])
        logp_sigma = standard_prior_dists['sigma'].log_prob(params[:, 4])
        logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma
    else:
        logp_z  = standard_prior_dists['z'].log_prob(params[:, 0])
        logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 1])
        logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 2])
        logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 3])
        logp_c  = standard_prior_dists['c'].log_prob(params[:, 4])
        logp_sigma = standard_prior_dists['sigma'].log_prob(params[:, 5])
        logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma
    return jnp.reshape(logp, (-1,))

@jax.jit
def compute_single_loglikelihood_standard(params):
    """Compute Gaussian log likelihood for a single set of parameters (standard)."""
    if fix_z:
        t0, log_x0, x1, c, sigma = params
        z = fixed_z[0]
    else:
        z, t0, log_x0, x1, c, sigma = params
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
        t0, log_x0, x1, c, sigma, log_p = params
        z = fixed_z[0]
    else:
        z, t0, log_x0, x1, c, sigma, log_p = params
    x0 = 10 ** log_x0
    p = 10 ** log_p  # Transform log10_p to p
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    eff_fluxerrs = sigma * fluxerrs
    point_logL = -0.5 * (((fluxes - model_fluxes) / eff_fluxerrs) ** 2) - 0.5 * jnp.log(2 * jnp.pi * eff_fluxerrs ** 2) + jnp.log(1 - p)
    delta = 250  # A threshold roughly corresponding to max data scale
    emax = point_logL > (log_p - jnp.log(delta))
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
            keys = jax.random.split(rng_key, 5)
            return jnp.column_stack([
                anomaly_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                anomaly_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                anomaly_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                anomaly_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                anomaly_prior_dists['sigma'].sample(seed=keys[4], sample_shape=(n_samples,)),
                anomaly_prior_dists['log_p'].sample(seed=keys[5], sample_shape=(n_samples,))
            ])
        else:
            keys = jax.random.split(rng_key, 6)
            return jnp.column_stack([
                anomaly_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                anomaly_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                anomaly_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                anomaly_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                anomaly_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                anomaly_prior_dists['sigma'].sample(seed=keys[5], sample_shape=(n_samples,)),
                anomaly_prior_dists['log_p'].sample(seed=keys[6], sample_shape=(n_samples,))
            ])
    else:  # Standard case
        if fix_z:
            keys = jax.random.split(rng_key, 5)
            return jnp.column_stack([
                standard_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                standard_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                standard_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                standard_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                standard_prior_dists['sigma'].sample(seed=keys[4], sample_shape=(n_samples,))
            ])
        else:
            keys = jax.random.split(rng_key, 6)
            return jnp.column_stack([
                standard_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                standard_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                standard_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                standard_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                standard_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                standard_prior_dists['sigma'].sample(seed=keys[5], sample_shape=(n_samples,))
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
def run_nested_sampling(ll_fn, output_prefix, num_iterations=NS_SETTINGS['max_iterations']):
    import os
    os.makedirs(output_prefix, exist_ok=True)
    print("Setting up nested sampling algorithm...")
    algo = blackjax.ns.adaptive.nss(
        logprior_fn=logprior_standard,
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
    print(f"Running nested sampling for {output_prefix}...")
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
        param_names = ['t0', 'log_x0', 'x1', 'c', 'sigma'] if fix_z else ['z', 't0', 'log_x0', 'x1', 'c', 'sigma']
    else:
        param_names = ['t0', 'log_x0', 'x1', 'c', 'sigma', 'log_p'] if fix_z else ['z', 't0', 'log_x0', 'x1', 'c', 'sigma', 'log_p']
    save_chains_dead_birth(dead, param_names)
    if ll_fn == loglikelihood_anomaly and emax_values:
        emax_array = jnp.stack(emax_values)
        print(f"emax_array shape: {emax_array.shape}")
        weights = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
        weights = weights[:, 0]
        weighted_emax = jnp.zeros(emax_array.shape[1])
        for i in range(emax_array.shape[1]):
            weighted_emax = weighted_emax.at[i].set(jnp.sum(emax_array[:, i] * weights) / jnp.sum(weights))
        print(f"weighted_emax shape: {weighted_emax.shape}")
        emax_output_path = os.path.join(output_prefix, f"{output_prefix}_weighted_emax.txt")
        np.savetxt(emax_output_path, weighted_emax)
        print(f"Saved weighted emax values to {emax_output_path}")

def get_n_params(ll_fn):
    if ll_fn == loglikelihood_standard:
        return 5 if fix_z else 6
    else:
        return 6 if fix_z else 7

if __name__ == "__main__":
    #print("Running standard version...")
    #n_params = get_n_params(loglikelihood_standard)
    #num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']
    #standard_samples = run_nested_sampling(loglikelihood_standard, "chains_standard")
    print("\nRunning anomaly detection version...")
    n_params = get_n_params(loglikelihood_anomaly)
    num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']
    anomaly_samples = run_nested_sampling(loglikelihood_anomaly, "chains_anomaly")