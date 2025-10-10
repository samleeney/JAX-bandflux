"""
# Nested Sampling with JAX-bandflux

This script demonstrates nested sampling for SALT3 supernova model fitting.
The redshift is fixed (not fitted) and flux uncertainties can optionally be scaled.
"""

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
from blackjax.ns.utils import log_weights
from jax_supernovae import SALT3Source
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_and_process_data
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes

# Configuration
fit_sigma = False  # Fit an extra parameter to scale flux uncertainties

NS_SETTINGS = {
    'n_delete': 60,
    'n_live': 125,
    'num_mcmc_steps_multiplier': 5
}

PRIOR_BOUNDS = {
    't0': (58000.0, 59000.0),
    'x0': (-5.0, -2.6),
    'x1': (-4.0, 4.0),
    'c': (-0.3, 0.3),
    'log_sigma': (-3.0, 1.0)
}

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load data with fixed redshift
times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z = \
    load_and_process_data('19dwz', data_dir='data', fix_z=True)

z = fixed_z[0]
source = SALT3Source()

# Setup priors
base_params = ['t0', 'x0', 'x1', 'c']
prior_params = base_params + (['log_sigma'] if fit_sigma else [])
prior_dists = {p: distrax.Uniform(low=PRIOR_BOUNDS[p][0], high=PRIOR_BOUNDS[p][1])
               for p in prior_params}

@jax.jit
def logprior(params):
    """Calculate log prior probability."""
    params = jnp.atleast_2d(params)
    logp_parts = jnp.stack([prior_dists[prior_params[i]].log_prob(params[:, i])
                            for i in range(len(prior_params))], axis=0)
    return jnp.sum(logp_parts, axis=0)

@jax.jit
def compute_single_loglikelihood(params):
    """Compute Gaussian log likelihood using SALT3Source v3.0 API."""
    params = jnp.atleast_1d(params)
    if params.ndim > 1:
        return jax.vmap(compute_single_loglikelihood)(params)

    # Parse parameters
    t0, log_x0, x1, c = params[:4]
    sigma = 10 ** params[4] if fit_sigma else 1.0
    x0 = 10 ** log_x0

    # Calculate rest-frame phases
    phases = (times - t0) / (1 + z)

    # Calculate model fluxes
    model_fluxes = source.bandflux(
        {'x0': x0, 'x1': x1, 'c': c}, None, phases, zp=zps, zpsys='ab',
        band_indices=band_indices, bridges=bridges, unique_bands=unique_bands
    )

    # Compute chi-squared
    eff_fluxerrs = sigma * fluxerrs
    chi2 = jnp.sum(((fluxes - model_fluxes) / eff_fluxerrs) ** 2)
    return -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * eff_fluxerrs ** 2)))

def sample_from_priors(rng_key, n_samples):
    """Sample from prior distributions."""
    keys = jax.random.split(rng_key, len(prior_params))
    samples = [prior_dists[prior_params[i]].sample(seed=keys[i], sample_shape=(n_samples,))
               for i in range(len(prior_params))]
    return jnp.column_stack(samples)

# Setup nested sampling parameters
n_params_total = len(prior_params)
num_mcmc_steps = n_params_total * NS_SETTINGS['num_mcmc_steps_multiplier']
param_names = ['t0', 'log_x0', 'x1', 'c'] + (['log_sigma'] if fit_sigma else [])

# Initialize nested sampling
algo = blackjax.nss(
    logprior_fn=logprior,
    loglikelihood_fn=compute_single_loglikelihood,
    num_inner_steps=num_mcmc_steps,
    num_delete=NS_SETTINGS['n_delete'],
)

rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key)
initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'])
state = algo.init(initial_particles)

@jax.jit
def one_step(carry, _):
    """One step of nested sampling."""
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

# Run nested sampling
print(f"Running nested sampling on {jax.devices()[0]}...")
dead = []
with tqdm.tqdm(desc="Dead points", unit=" points") as pbar:
    while state.logZ_live - state.logZ >= -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(NS_SETTINGS['n_delete'])

# Process results
dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
logw = log_weights(rng_key, dead)
logZs = jax.scipy.special.logsumexp(logw, axis=0)

print(f"\nEvidence: {logZs.mean():.2f} ± {logZs.std():.2f}")

# Save and visualize results
save_chains_dead_birth(dead, param_names)
samples = read_chains('chains/chains', columns=param_names)

# Create corner plot
fig, axes = make_2d_axes(param_names, figsize=(12, 12), facecolor='w')
samples.plot_2d(axes, alpha=0.9, label="posterior")
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)),
                       loc='lower center', ncols=2)
plt.suptitle(f'SALT3 Posterior (z={z:.4f})', y=1.02, fontsize=14)
plt.savefig('corner_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Print and save statistics
print("\nParameter Statistics:")
print("-" * 50)
print(f"{'Parameter':<12} {'Mean':>15} {'Std Dev':>15}")
print("-" * 50)
stats_lines = []
for param in param_names:
    mean, std = samples[param].mean(), samples[param].std()
    line = f"{param:<12} {mean:>15.6f} {std:>15.6f}"
    print(line)
    stats_lines.append(line)
print("-" * 50)

with open('parameter_statistics.txt', 'w') as f:
    f.write("Parameter Statistics:\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Parameter':<12} {'Mean':>15} {'Std Dev':>15}\n")
    f.write("-" * 50 + "\n")
    f.write("\n".join(stats_lines) + "\n")
    f.write("-" * 50 + "\n")