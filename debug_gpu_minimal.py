"""
Minimal script to debug JAX GPU functionality
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import traceback

# Print JAX device information
print("Available JAX devices:", jax.devices())
print("JAX version:", jax.__version__)

# Try different approaches to use GPU
try:
    # Approach 1: Simple array creation
    print("\nApproach 1: Simple array creation")
    x = jnp.ones((10, 10))
    print("Created array on", x.device)
    
    # Approach 2: Matrix multiplication with small arrays
    print("\nApproach 2: Matrix multiplication with small arrays")
    x_small = jnp.ones((10, 10))
    y_small = jnp.ones((10, 10))
    result_small = jnp.dot(x_small, y_small).block_until_ready()
    print("Small matrix multiplication completed on", result_small.device)
    
    # Approach 3: Explicitly place on GPU
    print("\nApproach 3: Explicitly place on GPU")
    gpu_device = jax.devices("gpu")[0]
    with jax.default_device(gpu_device):
        x_gpu = jnp.ones((10, 10))
        print("Created array explicitly on", x_gpu.device)
    
    # Approach 4: Simple JIT function
    print("\nApproach 4: Simple JIT function")
    @jax.jit
    def simple_function(x):
        return jnp.sum(x)
    
    result_jit = simple_function(jnp.ones((10, 10))).block_until_ready()
    print("JIT function executed on", jax.devices()[0])
    
    print("\nAll approaches completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()