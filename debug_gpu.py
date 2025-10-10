"""
Debug script to test JAX GPU functionality
"""

import jax
import jax.numpy as jnp
import numpy as np
import os

# Print JAX device information
print("Available JAX devices:", jax.devices())
print("JAX version:", jax.__version__)

# Test basic JAX operations on GPU
print("\nTesting basic JAX operations on GPU...")
x = jnp.ones((1000, 1000))
y = jnp.ones((1000, 1000))

# Force computation to happen on GPU
print("Starting matrix multiplication...")
result = jnp.dot(x, y).block_until_ready()
print("Matrix multiplication completed successfully")

# Test random number generation
print("\nTesting random number generation...")
key = jax.random.PRNGKey(0)
random_nums = jax.random.normal(key, (1000, 1000))
print("Random number generation completed successfully")

# Test JIT compilation
print("\nTesting JIT compilation...")
@jax.jit
def jitted_function(x, y):
    return jnp.dot(x, y)

print("Starting JIT compilation...")
result = jitted_function(x, y).block_until_ready()
print("JIT compilation completed successfully")

print("\nAll basic GPU tests passed successfully!")