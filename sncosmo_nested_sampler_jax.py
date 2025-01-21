import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import distrax
import tqdm
import os
from functools import partial
from blackjax.ns.utils import log_weights
from jax_supernovae.core import Bandpass, MODEL_BANDFLUX_SPACING
from jax_supernovae.salt3nir import salt3nir_bandflux, integration_grid

def ensure_chains_dir():
    """Ensure the chains directory exists."""
    os.makedirs('chains', exist_ok=True)

def save_chain_files(samples, weights, logw, logZs, param_names):
    """Save nested sampling chains to files."""
    ensure_chains_dir()
    
    print("\nArray shapes before saving:")
    print(f"samples: {samples.shape}")
    print(f"weights: {weights.shape}")
    print(f"logw: {logw.shape}")

    print("\nArray shapes before conversion:")
    print(f"samples: {samples.shape}")
    print(f"weights: {weights.shape}")
    print(f"logw: {logw.shape}")

    # Convert to numpy arrays and take only the first column
    weights = np.array(weights)[:, 0].reshape(-1, 1)
    logw = np.array(logw)[:, 0].reshape(-1, 1)
    samples = np.array(samples)

    print("\nArray shapes after conversion:")
    print(f"samples: {samples.shape}")
    print(f"weights: {weights.shape}")
    print(f"logw: {logw.shape}")

    physical_data = np.hstack([samples, weights, logw])
    np.savetxt('chains/chains_phys.txt', physical_data, header=' '.join(param_names) + ' weight logweight')
    
    # Save birth contours (just the samples)
    birth_data = samples
    np.savetxt('chains/chains_phys_birth.txt', birth_data, header=' '.join(param_names))
    
    # Save live points (same as birth for final iteration)
    np.savetxt('chains/chains_phys_live.txt', birth_data, header=' '.join(param_names))
    
    # Save live birth points (same as birth)
    np.savetxt('chains/chains_phys_live-birth.txt', birth_data, header=' '.join(param_names))
    
    # Save evidence
    print(f"\nLog-Evidence (logZ): {float(np.array(logZs)[0]):.2f}")

# Load the data
import sncosmo
data = sncosmo.load_example_data()
print(data)

# Convert bandpasses to JAX-compatible format once and set up integration grids
bandpasses = []
unique_bands = np.unique(data['band'])
band_to_idx = {band: i for i, band in enumerate(unique_bands)}
for band_name in unique_bands:
    # Get original bandpass
    snc_bandpass = sncosmo.get_bandpass(band_name)
    
    # Convert to JAX-compatible bandpass
    bandpass = Bandpass(snc_bandpass.wave, snc_bandpass.trans)
    bandpasses.append(bandpass)

# Convert data arrays to JAX arrays once
data_flux = jnp.array(data['flux'])
data_fluxerr = jnp.array(data['fluxerr'])
data_time = jnp.array(data['time'])

# Create array of bandpass indices
bandpass_indices = jnp.array([band_to_idx[band] for band in data['band']])

# Pre-compute phase indices for each bandpass
phase_indices = []
for i in range(len(unique_bands)):
    indices = np.where(bandpass_indices == i)[0]
    phase_indices.append(indices)

# Convert phase indices to JAX arrays
phase_indices = [jnp.array(idx) for idx in phase_indices]

# Model parameters: z, t0, x0, x1, c
param_names = ['z', 't0', 'x0', 'x1', 'c']

# Define parameter bounds
param_bounds = {
    'z': (0.3, 0.7),
    't0': (55080., 55120.),
    'x0': (1e-6, 1e-2),
    'x1': (-3., 3.),
    'c': (-0.3, 0.3)
}

@partial(jax.jit, static_argnames=['bandpass'])
def compute_single_bandpass_fluxes(params, phases, bandpass):
    """Compute fluxes for a single bandpass with batched parameters."""
    param_dict = {name: params[i] for i, name in enumerate(param_names)}
    return salt3nir_bandflux(phases, bandpass, param_dict)

@jax.jit
def compute_loglikelihood_for_particle(params):
    """Compute log-likelihood for a single particle."""
    # Calculate model fluxes for each unique bandpass
    model_fluxes = jnp.zeros_like(data_flux)
    for bandpass_idx in range(len(bandpasses)):
        # Get phases for this bandpass using pre-computed indices
        indices = phase_indices[bandpass_idx]
        phases = data_time[indices]
        
        # Calculate fluxes for all phases at once
        fluxes = compute_single_bandpass_fluxes(params, phases, bandpasses[bandpass_idx])
        
        # Update model_fluxes at the correct indices
        model_fluxes = model_fluxes.at[indices].set(fluxes)
    
    # Compute chi-squared
    chi2 = jnp.sum(((data_flux - model_fluxes) / data_fluxerr) ** 2)
    
    # Compute normalization term
    norm_term = jnp.sum(jnp.log(data_fluxerr * jnp.sqrt(2 * jnp.pi)))
    
    # Return normalized log-likelihood
    return -0.5 * chi2 - norm_term

@jax.jit
def loglikelihood(parameters):
    """Compute log-likelihood for a batch of particles."""
    return jax.vmap(compute_loglikelihood_for_particle)(parameters)

class UniformJointPrior(distrax.Distribution):
    def __init__(self, param_names, param_bounds):
        super().__init__()
        self._param_names = param_names
        self._param_bounds = param_bounds
        
    def _sample_n(self, key, n):
        """Sample n points from the joint prior"""
        keys = jax.random.split(key, len(self._param_names))
        samples = []
        for i, name in enumerate(self._param_names):
            low, high = self._param_bounds[name]
            samples.append(
                jax.random.uniform(
                    keys[i], 
                    shape=(n,), 
                    minval=low, 
                    maxval=high
                )
            )
        return jnp.stack(samples, axis=-1)
    
    def log_prob(self, x):
        """Log probability of the joint prior"""
        total_log_prob = 0.0
        for i, name in enumerate(self._param_names):
            low, high = self._param_bounds[name]
            # Uniform prior in the specified range
            is_in_bounds = (x[..., i] >= low) & (x[..., i] <= high)
            total_log_prob = jnp.where(
                is_in_bounds,
                total_log_prob - jnp.log(high - low),
                -jnp.inf
            )
        return total_log_prob
    
    @property
    def event_shape(self):
        """Shape of a single event"""
        return (len(self._param_names),)

# Create prior distribution
prior = UniformJointPrior(param_names, param_bounds)

# Nested sampling parameters
n_live = len(param_names) * 25
n_delete = 1
num_mcmc_steps = len(param_names) * 5

# Initialize the nested sampling algorithm
algo = blackjax.ns.adaptive.nss(
    logprior_fn=prior.log_prob,
    loglikelihood_fn=loglikelihood,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
)

# Random key for reproducibility
rng_key = jax.random.PRNGKey(42)

# Initialize with samples from prior
rng_key, init_key = jax.random.split(rng_key)
initial_particles = prior.sample(seed=init_key, sample_shape=(n_live,))
state = algo.init(initial_particles, loglikelihood)

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

# Run nested sampling
dead = []
for _ in tqdm.trange(10):
    if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        break
    (state, rng_key), dead_info = one_step((state, rng_key), None)
    dead.append(dead_info)

# Process results
dead = jax.tree_util.tree_map(lambda *args: jnp.concatenate(args), *dead)
print("\nDead points structure:")
print(jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, dead))
logw = log_weights(rng_key, dead)
logZs = jax.scipy.special.logsumexp(logw, axis=0)

# Extract samples and compute statistics
samples = dead.particles
weights = jnp.exp(logw - logZs)

# Print shapes for debugging
print("\nArray shapes before saving:")
print(f"samples: {samples.shape}")
print(f"weights: {weights.shape}")
print(f"logw: {logw.shape}")

# Save chains
save_chain_files(samples, weights, logw, logZs, param_names)

# Calculate mean and standard deviation for each parameter
weights_1d = weights[:, 0]  # Take only the first column
mean_params = jnp.average(samples, axis=0, weights=weights_1d)
std_params = jnp.sqrt(jnp.average((samples - mean_params) ** 2, axis=0, weights=weights_1d))

# Print parameter estimates
print("\nParameter Estimates:")
for i, name in enumerate(param_names):
    mean = mean_params[i]
    std = std_params[i]
    print(f"{name} = {mean:.5f} ± {std:.5f}")

print("\nChain files have been saved in the 'chains' directory.") 