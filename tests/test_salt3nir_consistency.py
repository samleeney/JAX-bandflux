import os
import sys

# Add project root to Python path if running the file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

import numpy as np
import jax.numpy as jnp
import sncosmo
from sncosmo.utils import integration_grid as sncosmo_integration_grid
from jax_supernovae.salt3 import (
    salt3_m0, salt3_m1, salt3_colorlaw,
    salt3_bandflux
)
from jax_supernovae.bandpasses import Bandpass, MODEL_BANDFLUX_SPACING

def test_salt3_consistency():
    """Test consistency between JAX implementation and sncosmo for SALT3-NIR model."""
    # Set model parameters
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create sncosmo model and set parameters
    snc_model = sncosmo.Model(source='salt3-nir')
    snc_model.update(params)

    # Test phase and wavelengths
    phase = 5.0
    wavelengths = np.linspace(3000.0, 9000.0, 100)

    # ------------------------------
    # Part (a): Test m0, m1, and c
    # ------------------------------

    # Get m0, m1, and color law from sncosmo
    snc_m0 = snc_model._source._model['M0'](np.array([phase]), wavelengths)[0]
    snc_m1 = snc_model._source._model['M1'](np.array([phase]), wavelengths)[0]
    snc_cl = snc_model._source._colorlaw(wavelengths)

    # Get m0, m1, and color law from JAX implementation
    jax_m0 = salt3_m0(phase, wavelengths)
    jax_m1 = salt3_m1(phase, wavelengths)
    jax_cl = salt3_colorlaw(wavelengths)

    # Assert that the components match
    np.testing.assert_allclose(jax_m0, snc_m0, rtol=1e-6,
                               err_msg="M0 components do not match")
    np.testing.assert_allclose(jax_m1, snc_m1, rtol=1e-6,
                               err_msg="M1 components do not match")
    np.testing.assert_allclose(jax_cl, snc_cl, rtol=1e-6,
                               err_msg="Color law components do not match")

    # --------------------------------
    # Part (b): Test Integration Grids
    # --------------------------------

    # Define integration grid parameters
    low = 3000.0
    high = 9000.0
    spacing = MODEL_BANDFLUX_SPACING

    # Get integration grid from sncosmo
    snc_wave, snc_dwave = sncosmo_integration_grid(low, high, spacing)

    # Create a test bandpass with the original wavelength range
    test_wave = np.linspace(low, high, 100)  # Original wavelength sampling
    test_bandpass = Bandpass(test_wave, np.ones_like(test_wave))

    # Print integration grid comparison
    print("\nIntegration Grid Comparison:")
    print("-" * 60)
    print(f"{'Index':>6} {'SNCosmo Wave':>15} {'JAX Wave':>15} {'Diff':>15}")
    print("-" * 60)
    for i in range(min(5, len(snc_wave))):  # Print first 5 points
        diff = float(test_bandpass.integration_wave[i] - snc_wave[i])
        print(f"{i:6d} {snc_wave[i]:15.6f} {float(test_bandpass.integration_wave[i]):15.6f} {diff:15.6e}")
    print("...")
    for i in range(-5, 0):  # Print last 5 points
        diff = float(test_bandpass.integration_wave[i] - snc_wave[i])
        print(f"{len(snc_wave)+i:6d} {snc_wave[i]:15.6f} {float(test_bandpass.integration_wave[i]):15.6f} {diff:15.6e}")
    print("-" * 60)

    # Assert that the integration grids match
    np.testing.assert_allclose(test_bandpass.integration_wave, snc_wave, rtol=1e-10,
                               err_msg="Integration grids do not match")

    # ---------------------------------
    # Part (c): Test Bandfluxes Match
    # ---------------------------------

    # Create a flat bandpass covering the wavelength range
    band_wave = np.linspace(low, high, int((high - low) / spacing) + 1)
    band_trans = np.ones_like(band_wave)
    snc_band = sncosmo.Bandpass(band_wave, band_trans, name='flat_band')
    jax_band = Bandpass(band_wave, band_trans)

    # Define test phases
    test_phases = [-10.0, 0.0, 10.0, 20.0]

    print("\nBandflux Comparison:")
    print("-" * 60)
    print(f"{'Phase':>8} {'SNCosmo Flux':>15} {'JAX Flux':>15} {'Ratio':>10}")
    print("-" * 60)

    # Test both with and without zeropoint scaling
    test_cases = [
        {'zp': None, 'zpsys': None},
        {'zp': 25.0, 'zpsys': 'ab'}
    ]

    for test_case in test_cases:
        print(f"\nTesting with zp={test_case['zp']}, zpsys={test_case['zpsys']}")
        print("-" * 60)

        for phase in test_phases:
            # Calculate bandflux using sncosmo
            snc_flux = snc_model.bandflux(snc_band, phase, 
                                        zp=test_case['zp'], 
                                        zpsys=test_case['zpsys'])

            # Calculate bandflux using JAX implementation
            jax_flux = salt3_bandflux(phase, jax_band, params,
                                       zp=test_case['zp'],
                                       zpsys=test_case['zpsys'])
            
            # Convert JAX array to scalar
            jax_flux = jnp.squeeze(jax_flux)

            # Print comparison
            ratio = float(jax_flux/snc_flux)
            print(f"{phase:8.1f} {snc_flux:15.6e} {float(jax_flux):15.6e} {ratio:10.4f}")

            # Assert that the bandfluxes match
            np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-2,
                                     err_msg=f"Bandfluxes do not match at phase {phase} "
                                            f"with zp={test_case['zp']}, zpsys={test_case['zpsys']}")

        print("-" * 60)

if __name__ == "__main__":
    test_salt3_consistency()
    print("\nAll tests passed.") 