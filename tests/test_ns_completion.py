import os
import sys
import pytest
import jax
import subprocess

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def test_ns_completion():
    """Test that nested sampling runs successfully for a small number of iterations.
    
    This test runs the nested sampling example script with a limited number of iterations
    to verify that the core functionality works correctly without running the full
    computation. It sets an environment variable to limit iterations, executes the
    script as a subprocess, and checks for successful completion.
    """
    
    try:
        # Set environment variable for max iterations
        os.environ['NS_MAX_ITERATIONS'] = '100'
        
        # Run the actual ns.py script
        script_path = os.path.join(project_root, 'examples/ns.py')
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        
        # Check if the script ran successfully
        if result.returncode != 0:
            pytest.fail(f"Script failed with error: {result.stderr}")
        
        # Test passed if we got here without errors
        assert True
        
    except Exception as e:
        pytest.fail(f"Nested sampling failed: {str(e)}")
        
    finally:
        # Clean up environment variable
        if 'NS_MAX_ITERATIONS' in os.environ:
            del os.environ['NS_MAX_ITERATIONS'] 