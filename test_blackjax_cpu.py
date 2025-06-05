"""
Test script to debug BlackJAX initialization on CPU
"""

import os
import sys
import traceback

# Force CPU usage
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import distrax
import blackjax

# Print JAX device information
print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("Default device:", jax.default_backend())

# Enable float64 precision
jax.config.update("jax_enable_x64", True)
print("jax_enable_x64:", jax.config.jax_enable_x64)

# Define a simple prior and likelihood
def simple_logprior(params):
    """Simple uniform prior on [-5, 5] for each parameter"""
    # Check if all parameters are within bounds
    in_bounds = jnp.all((params >= -5) & (params <= 5))
    # Return 0 if in bounds, -infinity if out of bounds
    return jnp.where(in_bounds, 0.0, -jnp.inf)

def simple_loglikelihood(params):
    """Simple Gaussian likelihood centered at the origin"""
    return -0.5 * jnp.sum(params**2)

# Try to initialize BlackJAX with minimal parameters
try:
    print("\nInitializing BlackJAX with minimal parameters...")
    
    # Create a simple algorithm with just 2 parameters
    n_params = 2
    n_live = 10
    num_mcmc_steps = 5
    
    # Initialize the algorithm
    print("Creating algorithm...")
    algo = blackjax.ns.adaptive.nss(
        logprior_fn=simple_logprior,
        loglikelihood_fn=simple_loglikelihood,
        n_delete=1,
        num_mcmc_steps=num_mcmc_steps,
    )
    print("Algorithm created successfully")
    
    # Generate initial particles
    print("Generating initial particles...")
    rng_key = jax.random.PRNGKey(0)
    initial_particles = jax.random.uniform(
        rng_key, 
        shape=(n_live, n_params), 
        minval=-5.0, 
        maxval=5.0
    )
    print(f"Initial particles generated with shape: {initial_particles.shape}")
    
    # Initialize the state
    print("Initializing algorithm state...")
    state = algo.init(initial_particles, simple_loglikelihood)
    print("Algorithm state initialized successfully")
    
    # Try a single step
    print("\nTrying a single step...")
    rng_key, step_key = jax.random.split(rng_key)
    state, dead_point = algo.step(step_key, state)
    print("Step completed successfully")
    
    print("\nAll tests passed successfully!")
    
except Exception as e:
    print(f"\nError: {e}")
    traceback.print_exc()