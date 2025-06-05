"""
Script to run ns.py with CPU only and additional debugging
"""

import os
import subprocess
import sys

# Force CPU usage
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Print environment information
print("Running with JAX_PLATFORM_NAME =", os.environ.get("JAX_PLATFORM_NAME"))

# Run the ns.py script with Python
print("\nRunning ns.py with CPU only...")
try:
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

print("\nScript completed.")