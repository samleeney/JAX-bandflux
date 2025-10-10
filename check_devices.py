"""
Simple script to check JAX devices without performing operations
"""

import jax
import os

# Print environment variables
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "Not set"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "Not set"))

# Print JAX configuration
print("\nJAX Configuration:")
print("JAX version:", jax.__version__)

# Try to get devices without performing operations
try:
    print("\nAttempting to list devices...")
    devices = jax.devices()
    print("Available JAX devices:", devices)
    
    # Print more details about each device
    for i, device in enumerate(devices):
        print(f"\nDevice {i} details:")
        print(f"  Device type: {device.device_kind}")
        print(f"  Device ID: {device.id}")
        print(f"  Platform: {device.platform}")
except Exception as e:
    print(f"Error listing devices: {e}")