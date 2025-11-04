"""
Nested sampling example for SALT3 parameter inference using JAX-bandflux.

This variant only covers the configuration used in the accompanying paper
and slides: the redshift is fixed to its measured heliocentric value and we
introduce a hyper-parameter `log_sigma` that rescales the reported flux errors
(`fit_sigma = True`). Removing the other branches keeps the example short
and easier to follow.
"""

import anesthetic
import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from blackjax.ns.utils import finalise, uniform_prior
from jax_supernovae import SALT3Source
from jax_supernovae.data import load_and_process_data

# Enable float64 precision for the model and nested sampling machinery.
jax.config.update("jax_enable_x64", True)

# Configuration constants
SUPERNOVA_ID = "19dwz"
DATA_DIR = "data"
FIXED_Z = 0.04607963148708845
NS_SETTINGS = {
    "n_delete": 60,
    "n_live": 125,
    "num_mcmc_steps_multiplier": 5,
}
PRIOR_BOUNDS = {
    "t0": (58000.0, 59000.0),
    "log_x0": (-5.0, -2.6),
    "x1": (-4.0, 4.0),
    "c": (-0.3, 0.3),
    "log_sigma": (-3.0, 1.0),
}

# Fixed parameter ordering used everywhere in the script.
PARAM_NAMES = ["t0", "log_x0", "x1", "c", "log_sigma"]
N_PARAMS = len(PARAM_NAMES)
NUM_MCMC_STEPS = N_PARAMS * NS_SETTINGS["num_mcmc_steps_multiplier"]

# Load and preprocess the photometric data.
times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, _ = load_and_process_data(
    SUPERNOVA_ID,
    data_dir=DATA_DIR,
    fix_z=False,
)

# Instantiate SALT3 source for bandflux calls.
source = SALT3Source()


@jax.jit
def loglikelihood(params: dict) -> jnp.ndarray:
    """Gaussian log-likelihood for the SALT3 light-curve model.

    Args:
        params: Dictionary containing t0, log_x0, x1, c, log_sigma.
                Each value can be a scalar or an array for vectorized evaluation.

    Returns:
        Log-likelihood value(s).
    """
    # Extract parameters from dictionary
    t0 = params["t0"]
    log_x0 = params["log_x0"]
    x1 = params["x1"]
    c = params["c"]
    log_sigma = params["log_sigma"]

    # Transform from log-space
    x0 = 10.0 ** log_x0
    sigma = 10.0 ** log_sigma

    # Compute model
    phases = (times - t0) / (1.0 + FIXED_Z)
    param_dict = {"x0": x0, "x1": x1, "c": c}

    model_fluxes = source.bandflux(
        param_dict,
        bands=None,
        phases=phases,
        zp=zps,
        zpsys="ab",
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands,
    )

    # Compute log-likelihood
    eff_fluxerrs = sigma * fluxerrs
    chi2 = jnp.sum(((fluxes - model_fluxes) / eff_fluxerrs) ** 2)
    log_det = jnp.sum(jnp.log(2.0 * jnp.pi * eff_fluxerrs**2))
    return -0.5 * (chi2 + log_det)


print("Setting up nested sampling...")
print(f"Using fixed heliocentric redshift z_hel = {FIXED_Z}")

# Initialize random key and create structured prior samples using uniform_prior
rng_key = jax.random.PRNGKey(0)
rng_key, prior_key = jax.random.split(rng_key)
particles, logprior_fn = uniform_prior(prior_key, NS_SETTINGS["n_live"], PRIOR_BOUNDS)

print(f"Particle structure: {particles.keys()}")
first_param = PARAM_NAMES[0]
print(f"Shape of each parameter: {particles[first_param].shape}")

# Initialize nested sampling algorithm
algo = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood,
    num_inner_steps=NUM_MCMC_STEPS,
    num_delete=NS_SETTINGS["n_delete"],
)

state = algo.init(particles)
print("Using device:", jax.devices()[0])

@jax.jit
def one_step(carry, _):
    state, key = carry
    key, subkey = jax.random.split(key)
    state, dead_point = algo.step(subkey, state)
    return (state, key), dead_point

dead_points = []
with tqdm.tqdm(desc="Dead points", unit=" dead") as progress:
    while not state.logZ_live - state.logZ < -3.0:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead_points.append(dead_info)
        progress.update(NS_SETTINGS["n_delete"])

ns_run = finalise(state, dead_points)

print("Building corner plot with anesthetic...")
# Create NestedSamples object with structured parameter dictionary
nested_samples = anesthetic.NestedSamples(
    data=ns_run.particles,
    logL=ns_run.loglikelihood,
    logL_birth=ns_run.loglikelihood_birth,
)

# Save samples to HDF5
nested_samples.to_csv("samples.csv")

# Create corner plot using parameter names
nested_samples.plot_2d(
    PARAM_NAMES,
    label="posterior",
)
plt.show()

