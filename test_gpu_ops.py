"""
Test script for JAX GPU operations
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import os

# Print JAX configuration
print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("Default device:", jax.default_backend())

# Try to force GPU usage
os.environ["JAX_PLATFORM_NAME"] = "gpu"
print("JAX_PLATFORM_NAME:", os.environ.get("JAX_PLATFORM_NAME", "Not set"))

# Test 1: Simple array creation
print("\nTest 1: Simple array creation")
try:
    x = jnp.ones((10, 10))
    print("Created array shape:", x.shape)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Very small matrix multiplication
print("\nTest 2: Very small matrix multiplication")
try:
    x_small = jnp.ones((2, 2))
    y_small = jnp.ones((2, 2))
    start_time = time.time()
    result_small = jnp.dot(x_small, y_small)
    # Force computation to complete
    result_small.block_until_ready()
    end_time = time.time()
    print(f"Small matrix multiplication completed in {end_time - start_time:.6f} seconds")
    print("Result shape:", result_small.shape)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Simple JIT function
print("\nTest 3: Simple JIT function")
try:
    @jax.jit
    def add_one(x):
        return x + 1
    
    input_array = jnp.zeros((2, 2))
    start_time = time.time()
    result = add_one(input_array)
    # Force computation to complete
    result.block_until_ready()
    end_time = time.time()
    print(f"JIT function completed in {end_time - start_time:.6f} seconds")
    print("Result shape:", result.shape)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

print("\nAll tests completed")