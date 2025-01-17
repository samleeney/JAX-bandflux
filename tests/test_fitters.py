import pytest
import numpy as np
import subprocess
import sys
import os
import sncosmo

def run_python_script(script_path):
    """Run a Python script and capture its output."""
    # Use the virtual environment Python
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_exe = os.path.join(workspace_root, "venv", "bin", "python")
    
    # Set up environment with correct Python path
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = workspace_root
    
    try:
        result = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=workspace_root  # Run from workspace root
        )
        
        # Print output for debugging
        print(f"\nOutput from {script_path}:")
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
            
        # Find the line starting with "RESULT:" and parse the parameters
        for line in result.stdout.strip().split('\n'):
            if line.startswith("RESULT:"):
                params_str = line.replace("RESULT:", "").strip()
                return np.array(eval(params_str))
                
        raise ValueError(f"No RESULT line found in output from {script_path}:\nStdout: {result.stdout}\nStderr: {result.stderr}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        raise

def compute_chi2(parameters):
    """Compute chi-squared for a set of parameters"""
    model = sncosmo.Model(source='salt2')
    data = sncosmo.load_example_data()
    
    model.parameters = parameters
    model_flux = model.bandflux(data['band'], data['time'],
                               zp=data['zp'], zpsys=data['zpsys'])
    
    return np.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)

def test_fitter_consistency():
    """Test that sncosmo-fitter.py and sncosmo-fitter-jax.py produce consistent results."""
    # Get paths relative to this test file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_fitter = os.path.join(base_dir, "sncosmo-fitter.py")
    jax_fitter = os.path.join(base_dir, "sncosmo-fitter-jax.py")
    
    print("\nRunning original fitter...")
    original_params = run_python_script(original_fitter)
    print("\nRunning JAX fitter...")
    jax_params = run_python_script(jax_fitter)
    
    # Compute chi-squared values
    original_chi2 = compute_chi2(original_params)
    jax_chi2 = compute_chi2(jax_params)
    
    # Print parameters and chi-squared values
    param_names = ['z', 't0', 'x0', 'x1', 'c']
    print("\nParameter comparison:")
    print(f"{'Parameter':<10} {'Original':<15} {'JAX':<15} {'Difference':<15}")
    print("-" * 55)
    for name, orig, jax in zip(param_names, original_params, jax_params):
        diff = abs(orig - jax)
        print(f"{name:<10} {orig:<15.6g} {jax:<15.6g} {diff:<15.6g}")
    
    print(f"\nChi-squared values:")
    print(f"Original: {original_chi2:.6f}")
    print(f"JAX:      {jax_chi2:.6f}")
    print(f"Relative difference: {abs(original_chi2 - jax_chi2) / original_chi2:.6f}")
    
    # Verify that JAX version finds a solution at least as good as original
    assert jax_chi2 <= original_chi2 * (1 + 1e-4), \
        f"JAX version found a worse solution:\nOriginal chi^2: {original_chi2}\nJAX chi^2: {jax_chi2}" 