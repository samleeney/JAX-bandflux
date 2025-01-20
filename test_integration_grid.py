import numpy as np
import jax.numpy as jnp
from sncosmo.utils import integration_grid as sncosmo_integration_grid
from jax_supernovae.salt3nir import integration_grid as jax_integration_grid

def test_integration_grid():
    # Define test parameters
    test_cases = [
        (4000.0, 8000.0, 5.0),
        (3000.0, 9000.0, 10.0),
        (3500.0, 7500.0, 2.5),
        (5000.0, 10000.0, 1.0)
    ]

    for i, (low, high, spacing) in enumerate(test_cases):
        # Get grids and spacings from both implementations
        sncosmo_wave, sncosmo_spacing = sncosmo_integration_grid(low, high, spacing)
        jax_wave, jax_spacing = jax_integration_grid(low, high, spacing)

        # Compare the results
        assert np.allclose(sncosmo_wave, np.array(jax_wave), rtol=1e-10), \
            f"Wave grids do not match for test case {i}: {sncosmo_wave} vs {jax_wave}"
        assert np.isclose(sncosmo_spacing, jax_spacing, rtol=1e-10), \
            f"Spacings do not match for test case {i}: {sncosmo_spacing} vs {jax_spacing}"

        print(f"Test case {i} passed: wave grids and spacings match.")

if __name__ == "__main__":
    test_integration_grid()