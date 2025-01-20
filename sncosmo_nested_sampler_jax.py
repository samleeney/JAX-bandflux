import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import distrax
import tqdm
from blackjax.ns.utils import log_weights
from jax_supernovae.models import Model
from jax_supernovae.core import get_magsystem
from archive.bandpasses import get_bandpass
from jax_supernovae.salt2 import salt2_flux
from jax_supernovae.salt2_data import get_salt2_wave_grid

# Load the data
import sncosmo
data = sncosmo.load_example_data()

# Initialize JAX model
jax_model = Model()
# Get wavelength grid
jax_model.wave = get_salt2_wave_grid()
jax_model.flux = lambda t, w: salt2_flux(t, w, jax_model.parameters)

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

def loglikelihood(parameters):
    """Compute the log-likelihood under Gaussian errors."""
    # Handle both single and batched parameters
    if parameters.ndim == 1:
        # Single set of parameters
        param_dict = {name: parameters[i] for i, name in enumerate(param_names)}
        jax_model.parameters = param_dict
        model_flux = jax_model.bandflux(
            data['band'], data['time'], zp=data['zp'], zpsys=data['zpsys']
        )
    else:
        # Batched parameters
        def process_single(p):
            param_dict = {name: p[i] for i, name in enumerate(param_names)}
            jax_model.parameters = param_dict
            return jax_model.bandflux(
                data['band'], data['time'], zp=data['zp'], zpsys=data['zpsys']
            )
        model_flux = jax.vmap(process_single)(parameters)
    
    # Convert to JAX arrays
    jax_model_flux = jnp.array(model_flux)
    jax_data_flux = jnp.array(data['flux'])
    jax_fluxerr = jnp.array(data['fluxerr'])
    
    # Compute chi-squared
    chi2 = jnp.sum(((jax_data_flux - jax_model_flux) / jax_fluxerr) ** 2, axis=-1)
    # Return log-likelihood
    return -0.5 * chi2

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
n_live = 500
n_delete = 20
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
for _ in tqdm.trange(1000):
    if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        break
    (state, rng_key), dead_info = one_step((state, rng_key), None)
    dead.append(dead_info)

# Process results
dead = jax.tree_util.tree_map(lambda *args: jnp.concatenate(args), *dead)
logw = log_weights(rng_key, dead)
logZs = jax.scipy.special.logsumexp(logw, axis=0)

# Extract samples and compute statistics
samples = dead.position
weights = jnp.exp(logw - logZs)

# Compute weighted mean and covariance
mean_params = jnp.average(samples, axis=0, weights=weights)
cov_params = jnp.cov(samples, rowvar=False, aweights=weights)

print("\nEstimated parameters:")
for i, name in enumerate(param_names):
    mean = mean_params[i]
    std = jnp.sqrt(cov_params[i, i])
    print(f"{name} = {mean:.5f} ± {std:.5f}")

print(f"\nLog-Evidence (logZ): {logZs:.2f}") 