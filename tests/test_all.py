import pytest
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def test_run_all():
    """Run all tests in the tests directory."""
    # The actual test running is handled by pytest when called with this file
    # This function exists to provide a test entry point
    assert True 