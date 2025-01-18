import pytest
import jax
import jax.numpy as jnp
import numpy as np
import sncosmo
from sncosmo_nested_sampler_jax import (
    loglikelihood, prior, param_bounds, param_names,
    algo, n_live, n_delete, num_mcmc_steps,
    jax_model, salt2_flux
)

def test_prior_sampling():
    """Test that prior sampling works and respects bounds"""
    rng_key = jax.random.PRNGKey(0)
    samples = prior.sample(seed=rng_key, sample_shape=(1000,))
    
    # Check that samples are within bounds
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        assert jnp.all(samples[:, i] >= low)
        assert jnp.all(samples[:, i] <= high)

def test_salt2_flux():
    """Test that our JAX SALT2 flux implementation matches sncosmo"""
    # Set up test parameters
    param_array = jnp.array([0.5, 55100., 1e-5, 0.0, 0.0])
    param_dict = {name: param_array[i] for i, name in enumerate(param_names)}
    time = jnp.array([55100.0])
    wave = jnp.array([4000.0])
    
    # Compute flux with our implementation
    jax_flux = salt2_flux(time, wave, param_dict)
    
    # Compute flux with sncosmo
    model = sncosmo.Model(source='salt2')
    model.parameters = param_array
    sncosmo_flux = model.flux(time[0], wave[0])
    
    # Compare results
    assert jnp.allclose(jax_flux[0,0], sncosmo_flux, rtol=1e-5)

def test_loglikelihood():
    """Test that loglikelihood computation works with known parameters"""
    # Use a known parameter set that should give reasonable likelihood
    parameters = jnp.array([0.5, 55100., 1e-5, 0.0, 0.0])
    logL = loglikelihood(parameters)
    
    # Check basic properties
    assert jnp.isfinite(logL)
    assert isinstance(logL, jnp.ndarray)
    
    # Test that obviously bad parameters give worse likelihood
    bad_parameters = jnp.array([0.5, 0.0, 1e-5, 0.0, 0.0])  # unreasonable t0
    bad_logL = loglikelihood(bad_parameters)
    assert bad_logL < logL

def test_nested_sampler_initialization():
    """Test that nested sampler initializes correctly"""
    rng_key = jax.random.PRNGKey(42)
    initial_particles = prior.sample(seed=rng_key, sample_shape=(n_live,))
    state = algo.init(initial_particles, loglikelihood)
    
    # Check that state has expected attributes
    assert hasattr(state, 'sampler_state')
    assert hasattr(state.sampler_state, 'logZ')
    assert hasattr(state.sampler_state, 'logZ_live')
    
    # Check dimensions
    assert state.position.shape == (n_live, len(param_names))

def test_nested_sampler_step():
    """Test that nested sampler can take steps"""
    # Initialize
    rng_key = jax.random.PRNGKey(42)
    initial_particles = prior.sample(seed=rng_key, sample_shape=(n_live,))
    state = algo.init(initial_particles, loglikelihood)
    
    # Take a step
    rng_key, step_key = jax.random.split(rng_key)
    new_state, info = algo.step(step_key, state)
    
    # Check that state updates correctly
    assert new_state.sampler_state.logZ != state.sampler_state.logZ
    assert new_state.position.shape == state.position.shape
    
    # Check info contains expected fields
    assert hasattr(info, 'position')
    assert hasattr(info, 'loglikelihood')
    assert info.position.shape == (n_delete, len(param_names)) 