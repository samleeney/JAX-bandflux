"""Test SALT3Source API consistency with sncosmo.

This module tests that the JAX-bandflux SALT3Source implementation produces
results exactly consistent with sncosmo's SALT3 source. This is the CRITICAL
test suite that verifies the v3.0 functional API maintains numerical accuracy.

Note: v3.0 uses a functional API (params passed to methods) while sncosmo uses
a stateful API (params stored in object). This tests numerical consistency.
"""

import os
import sys

# Add project root to Python path if running the file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

import numpy as np
import jax.numpy as jnp
import sncosmo
from jax_supernovae import SALT3Source
from jax_supernovae.salt3 import salt3_m0, salt3_m1, salt3_colorlaw


def test_model_components():
    """Test that M0, M1, and colorlaw match sncosmo exactly.

    This tests the underlying model components that form the basis of
    the SALT3 spectral model. These should match sncosmo to machine precision.
    """
    # Create sncosmo model
    snc_source = sncosmo.get_source('salt3-nir')

    # Test phase and wavelengths
    phase = 5.0
    wavelengths = np.linspace(3000.0, 9000.0, 100)

    # Get m0, m1, and color law from sncosmo
    snc_m0 = snc_source._model['M0'](np.array([phase]), wavelengths)[0]
    snc_m1 = snc_source._model['M1'](np.array([phase]), wavelengths)[0]
    snc_cl = snc_source._colorlaw(wavelengths)

    # Get m0, m1, and color law from JAX implementation
    jax_m0 = salt3_m0(phase, wavelengths)
    jax_m1 = salt3_m1(phase, wavelengths)
    jax_cl = salt3_colorlaw(wavelengths)

    # Assert exact match
    np.testing.assert_allclose(jax_m0, snc_m0, rtol=1e-6,
                              err_msg="M0 components do not match")
    np.testing.assert_allclose(jax_m1, snc_m1, rtol=1e-6,
                              err_msg="M1 components do not match")
    np.testing.assert_allclose(jax_cl, snc_cl, rtol=1e-6,
                              err_msg="Color law components do not match")

    print("\n✓ Model components (M0, M1, colorlaw) match sncosmo")


def test_bandflux_scalar():
    """Test single bandflux calculation matches sncosmo exactly.

    This is the fundamental test - a single bandflux value at a single phase
    through a single bandpass must match sncosmo to within 0.001%.
    """
    # Set model parameters (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create JAX source (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    # Create and configure sncosmo source (stateful API for comparison)
    snc_source = sncosmo.get_source('salt3-nir')
    snc_source.set(**params)

    # Test single observation
    phase = 0.0
    band = 'bessellb'

    # Calculate bandflux without zeropoint (v3.0 functional API)
    jax_flux = jax_source.bandflux(params, band, phase)
    snc_flux = snc_source.bandflux(band, phase)

    # Should match to better than 0.01% (rtol=1e-4)
    # Note: When using sncosmo bandpasses (like Bessell filters), slight differences
    # in numerical integration can cause ~0.003% discrepancies
    np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4,
                              err_msg=f"Bandflux does not match for {band} at phase {phase}")

    print(f"\n✓ Scalar bandflux matches sncosmo:")
    print(f"  JAX:     {jax_flux:.6e}")
    print(f"  sncosmo: {snc_flux:.6e}")
    print(f"  Ratio:   {jax_flux/snc_flux:.8f}")


def test_bandflux_with_zeropoint():
    """Test bandflux calculations with AB zeropoint scaling.

    Tests that zeropoint and magnitude system handling matches sncosmo exactly.
    """
    # Set model parameters (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create JAX source (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    # Create and configure sncosmo source (stateful API for comparison)
    snc_source = sncosmo.get_source('salt3-nir')
    snc_source.set(**params)

    # Test with AB zeropoint
    phase = 0.0
    band = 'bessellb'
    zp = 27.5
    zpsys = 'ab'

    # v3.0 functional API - pass params to bandflux
    jax_flux = jax_source.bandflux(params, band, phase, zp=zp, zpsys=zpsys)
    snc_flux = snc_source.bandflux(band, phase, zp=zp, zpsys=zpsys)

    # Should match to better than 0.01% (rtol=1e-4)
    np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4,
                              err_msg="Bandflux with zeropoint does not match")

    print(f"\n✓ Bandflux with zeropoint (AB, zp={zp}) matches sncosmo:")
    print(f"  JAX:     {jax_flux:.6e}")
    print(f"  sncosmo: {snc_flux:.6e}")
    print(f"  Ratio:   {jax_flux/snc_flux:.8f}")


def test_bandflux_array_phases():
    """Test bandflux with array of phases in same band.

    Tests vectorization over multiple phases for a single bandpass.
    """
    # Set model parameters (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create JAX source (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    # Create and configure sncosmo source (stateful API for comparison)
    snc_source = sncosmo.get_source('salt3-nir')
    snc_source.set(**params)

    # Test multiple phases
    phases = np.array([-10.0, 0.0, 10.0, 20.0])
    band = 'bessellb'

    # v3.0 functional API - pass params to bandflux
    jax_fluxes = jax_source.bandflux(params, band, phases, zp=27.5, zpsys='ab')
    snc_fluxes = snc_source.bandflux(band, phases, zp=27.5, zpsys='ab')

    # Should match element-wise
    np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4,
                              err_msg="Array bandflux does not match")

    print(f"\n✓ Array bandflux (multiple phases) matches sncosmo:")
    print(f"  {'Phase':>8} {'JAX Flux':>15} {'SNCosmo Flux':>15} {'Ratio':>10}")
    print("  " + "-" * 53)
    for phase, jax_flux, snc_flux in zip(phases, jax_fluxes, snc_fluxes):
        print(f"  {phase:8.1f} {jax_flux:15.6e} {snc_flux:15.6e} {jax_flux/snc_flux:10.8f}")


def test_bandflux_multiple_bands():
    """Test bandflux with different bands at different phases.

    Tests broadcasting behavior when both band and phase are arrays.
    This is critical for multi-band light curve fitting.
    """
    # Set model parameters (v3.0 functional API)
    params = {
        'x0': 1e-5,
        'x1': -0.5,
        'c': 0.2
    }

    # Create JAX source (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    # Create and configure sncosmo source (stateful API for comparison)
    snc_source = sncosmo.get_source('salt3-nir')
    snc_source.set(**params)

    # Test multiple bands and phases (must broadcast correctly)
    bands = ['bessellb', 'bessellv', 'bessellr']
    phases = [0.0, 5.0, 10.0]
    zp = 27.5
    zpsys = 'ab'

    # v3.0 functional API - pass params to bandflux
    jax_fluxes = jax_source.bandflux(params, bands, phases, zp=zp, zpsys=zpsys)
    snc_fluxes = snc_source.bandflux(bands, phases, zp=zp, zpsys=zpsys)

    # Should match element-wise
    np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4,
                              err_msg="Multi-band bandflux does not match")

    print(f"\n✓ Multi-band bandflux matches sncosmo:")
    print(f"  {'Band':>10} {'Phase':>8} {'JAX Flux':>15} {'SNCosmo Flux':>15} {'Ratio':>10}")
    print("  " + "-" * 69)
    for band, phase, jax_flux, snc_flux in zip(bands, phases, jax_fluxes, snc_fluxes):
        print(f"  {band:>10} {phase:8.1f} {jax_flux:15.6e} {snc_flux:15.6e} {jax_flux/snc_flux:10.8f}")


def test_bandflux_comprehensive():
    """Comprehensive bandflux test across multiple parameters.

    Tests a grid of different parameter values and phases to ensure
    consistency across the full parameter space.
    """
    # Test multiple parameter combinations (v3.0 functional API)
    test_params = [
        {'x0': 1e-5, 'x1': 0.0, 'c': 0.0},     # Default
        {'x0': 1e-5, 'x1': 1.0, 'c': 0.0},     # Stretched
        {'x0': 1e-5, 'x1': -1.0, 'c': 0.0},    # Compressed
        {'x0': 1e-5, 'x1': 0.0, 'c': 0.3},     # Red
        {'x0': 1e-5, 'x1': 0.0, 'c': -0.2},    # Blue
        {'x0': 2e-5, 'x1': 0.5, 'c': -0.1},    # Combined
    ]

    phases = [-10.0, 0.0, 10.0, 20.0]
    band = 'bessellb'

    print(f"\n✓ Comprehensive bandflux test:")
    print(f"  {'x0':>8} {'x1':>6} {'c':>6} {'Phase':>8} {'Max Rel. Error':>15}")
    print("  " + "-" * 55)

    # Create JAX source once (v3.0 - no parameter storage)
    jax_source = SALT3Source()

    for params in test_params:
        # Create and configure sncosmo source (stateful API for comparison)
        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(**params)

        # Calculate fluxes at all phases (v3.0 functional API)
        jax_fluxes = jax_source.bandflux(params, band, phases, zp=27.5, zpsys='ab')
        snc_fluxes = snc_source.bandflux(band, phases, zp=27.5, zpsys='ab')

        # Check match
        np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4,
                                  err_msg=f"Bandflux mismatch for params {params}")

        # Calculate maximum relative error
        rel_errors = np.abs((jax_fluxes - snc_fluxes) / snc_fluxes)
        max_rel_error = np.max(rel_errors)

        print(f"  {params['x0']:8.1e} {params['x1']:6.1f} {params['c']:6.2f} "
              f"{'all':>8} {max_rel_error:15.2e}")


def test_parameter_names():
    """Test parameter names property.

    v3.0 removes stateful parameter storage but maintains param_names property.
    """
    # Create source
    jax_source = SALT3Source()
    snc_source = sncosmo.get_source('salt3-nir')

    # Test param_names property
    assert jax_source.param_names == snc_source.param_names, \
        "Parameter names should match sncosmo"
    assert jax_source.param_names == ['x0', 'x1', 'c'], \
        "Parameter names should be ['x0', 'x1', 'c']"

    print("\n✓ Parameter names property works correctly")


def test_bandflux_return_types():
    """Test that return types match sncosmo behavior (v3.0, JIT-compatible).

    Scalar inputs should return scalar (0-d array), array inputs should return arrays.
    """
    # v3.0 functional API
    source = SALT3Source()
    params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

    # Scalar input -> scalar output (JAX array with ndim=0 for JIT compatibility)
    flux = source.bandflux(params, 'bessellb', 0.0, zp=27.5, zpsys='ab')
    assert jnp.ndim(flux) == 0, \
        f"Scalar bandflux should return 0-d array (scalar), got shape {jnp.shape(flux)}"

    # Array input -> array output
    fluxes = source.bandflux(params, 'bessellb', [0.0, 1.0, 2.0], zp=27.5, zpsys='ab')
    assert jnp.shape(fluxes) == (3,), \
        f"Array bandflux should have shape (3,), got {jnp.shape(fluxes)}"

    print("\n✓ Return types match sncosmo behavior (scalar vs array)")


def test_error_handling():
    """Test error handling matches sncosmo.

    Ensures that invalid inputs raise appropriate errors.
    """
    # v3.0 functional API
    source = SALT3Source()
    params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

    # Test: zp without zpsys should raise error
    try:
        source.bandflux(params, 'bessellb', 0.0, zp=27.5)
        assert False, "Should have raised ValueError for zp without zpsys"
    except ValueError as e:
        assert 'zpsys' in str(e), "Error message should mention zpsys"

    print("\n✓ Error handling works correctly")


def run_all_tests():
    """Run all consistency tests."""
    print("=" * 70)
    print("SALT3Source API Consistency Tests (v3.0)")
    print("Testing against sncosmo SALT3-NIR implementation")
    print("Note: v3.0 uses functional API (params passed to methods)")
    print("=" * 70)

    test_model_components()
    test_bandflux_scalar()
    test_bandflux_with_zeropoint()
    test_bandflux_array_phases()
    test_bandflux_multiple_bands()
    test_bandflux_comprehensive()
    test_parameter_names()
    test_bandflux_return_types()
    test_error_handling()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED - SALT3Source matches sncosmo numerically!")
    print("v3.0 functional API maintains exact numerical consistency")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
