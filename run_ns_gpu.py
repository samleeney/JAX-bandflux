"""
Script to run ns.py with GPU and additional error handling
"""

import os
import subprocess
import sys
import traceback

# Force GPU usage
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# Print environment information
print("Running with JAX_PLATFORM_NAME =", os.environ.get("JAX_PLATFORM_NAME"))

# Import JAX to check if GPU is available
try:
    import jax
    print("JAX version:", jax.__version__)
    print("Available devices:", jax.devices())
    print("Default device:", jax.default_backend())
except Exception as e:
    print(f"Error importing JAX: {e}")
    traceback.print_exc()
    sys.exit(1)

# Run the ns.py script with Python
print("\nRunning ns.py with GPU...")
try:
    # Use a smaller number of live points to reduce memory usage
    os.environ["NS_LIVE_POINTS"] = "50"  # Default is 125
    os.environ["NS_MCMC_STEPS_MULTIPLIER"] = "3"  # Default is 5
    
    # Run the script
    result = subprocess.run(
        [sys.executable, "examples/ns.py"],
        check=True,
        capture_output=True,
        text=True
    )
    print("\nOutput from ns.py:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("\nError running ns.py:")
    print("Return code:", e.returncode)
    print("Standard output:", e.stdout)
    print("Standard error:", e.stderr)
except Exception as e:
    print("\nUnexpected error:", e)
    traceback.print_exc()

print("\nScript completed.")