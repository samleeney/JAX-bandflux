import os
import sys
import pytest
import jax
import yaml

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def test_ns_completion():
    """Test that nested sampling runs successfully for 100 iterations."""
    
    # First modify settings to run only 100 iterations
    with open('settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
    
    # Store original max_iterations
    original_max_iterations = settings['nested_sampling']['max_iterations']
    
    # Modify settings for test
    settings['nested_sampling']['max_iterations'] = 100
    
    # Write modified settings
    with open('settings.yaml', 'w') as f:
        yaml.dump(settings, f)
    
    try:
        # Import and run ns.py
        import ns
        
        # Test passed if we got here without errors
        assert True
        
    except Exception as e:
        pytest.fail(f"Nested sampling failed: {str(e)}")
        
    finally:
        # Restore original settings
        settings['nested_sampling']['max_iterations'] = original_max_iterations
        with open('settings.yaml', 'w') as f:
            yaml.dump(settings, f) 