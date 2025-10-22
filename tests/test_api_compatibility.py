"""Comprehensive behavior tests for SALT3Source v3.0 functional API.

This module provides extensive testing of the SALT3Source v3.0 functional API
to ensure it produces results consistent with sncosmo while using a functional
interface. This includes broadcasting behavior, return types, and error handling.

Note: v3.0 uses a functional API (params passed to methods) rather than sncosmo's
stateful API (params stored in object). These tests verify numerical consistency
and expected behaviors with the functional API.
"""

import os
import sys

# Add project root to Python path if running the file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

import numpy as np
import pytest
import sncosmo
from jax_supernovae import SALT3Source


class TestParameterNames:
    """Test parameter names property (v3.0 removes stateful parameter access)."""

    def test_param_names(self):
        """Test param_names property matches sncosmo."""
        jax_source = SALT3Source()
        snc_source = sncosmo.get_source('salt3-nir')

        # Should have same parameter names
        assert jax_source.param_names == snc_source.param_names
        assert jax_source.param_names == ['x0', 'x1', 'c']


class TestReturnTypes:
    """Test that return types match sncosmo exactly."""

    def test_scalar_phase_scalar_band_returns_float(self):
        """Scalar inputs should return scalar (v3.0 functional API, JIT-compatible)."""
        import jax.numpy as jnp

        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

        # Single band, single phase -> scalar (v3.0 functional API)
        flux = jax_source.bandflux(params, 'bessellb', 0.0, zp=27.5, zpsys='ab')
        # For JIT compatibility, returns JAX array with ndim=0 (scalar)
        assert jnp.ndim(flux) == 0, \
            f"Expected scalar (0-d array), got shape {jnp.shape(flux)}"

    def test_array_phase_returns_array(self):
        """Array of phases should return array (v3.0 functional API)."""
        import jax.numpy as jnp

        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

        # Single band, multiple phases -> array (v3.0 functional API)
        fluxes = jax_source.bandflux(params, 'bessellb', [0.0, 1.0, 2.0], zp=27.5, zpsys='ab')
        assert jnp.shape(fluxes) == (3,), f"Expected shape (3,), got {jnp.shape(fluxes)}"

    def test_array_band_returns_array(self):
        """Array of bands should return array (v3.0 functional API)."""
        import jax.numpy as jnp

        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

        # Multiple bands, single phase -> array (v3.0 functional API)
        fluxes = jax_source.bandflux(params, ['bessellb', 'bessellv'], 0.0, zp=27.5, zpsys='ab')
        assert jnp.shape(fluxes) == (2,), f"Expected shape (2,), got {jnp.shape(fluxes)}"

    def test_return_type_matches_sncosmo(self):
        """Return type should match sncosmo behavior (v3.0 functional API)."""
        import jax.numpy as jnp

        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        test_cases = [
            ('bessellb', 0.0),  # Scalar, scalar
            ('bessellb', [0.0, 1.0, 2.0]),  # Scalar, array
            (['bessellb', 'bessellv'], 0.0),  # Array, scalar
            (['bessellb', 'bessellv'], [0.0, 1.0]),  # Array, array
        ]

        for band, phase in test_cases:
            # v3.0 functional API - pass params
            jax_flux = jax_source.bandflux(params, band, phase, zp=27.5, zpsys='ab')
            snc_flux = snc_source.bandflux(band, phase, zp=27.5, zpsys='ab')

            # Check scalar vs array behavior by shape
            jax_is_scalar = jnp.ndim(jax_flux) == 0
            snc_is_scalar = isinstance(snc_flux, (float, np.floating)) and not isinstance(snc_flux, np.ndarray)

            assert jax_is_scalar == snc_is_scalar, \
                f"Scalar/array mismatch for band={band}, phase={phase}"

            # If array, shapes should match
            if not jax_is_scalar:
                assert jnp.shape(jax_flux) == snc_flux.shape, \
                    f"Shape mismatch for band={band}, phase={phase}"

            # Values should match
            np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4)


class TestBroadcasting:
    """Test broadcasting behavior matches sncosmo exactly."""

    def test_broadcast_same_band_multiple_phases(self):
        """Broadcasting: one band to multiple phases (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        band = 'bessellb'
        phases = [0.0, 5.0, 10.0]

        # v3.0 functional API - pass params
        jax_fluxes = jax_source.bandflux(params, band, phases, zp=27.5, zpsys='ab')
        snc_fluxes = snc_source.bandflux(band, phases, zp=27.5, zpsys='ab')

        # Shape should match
        assert jax_fluxes.shape == snc_fluxes.shape == (3,)

        # Values should match
        np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4)

    def test_broadcast_same_phase_multiple_bands(self):
        """Broadcasting: one phase to multiple bands (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        bands = ['bessellb', 'bessellv', 'bessellr']
        phase = 0.0

        # v3.0 functional API - pass params
        jax_fluxes = jax_source.bandflux(params, bands, phase, zp=27.5, zpsys='ab')
        snc_fluxes = snc_source.bandflux(bands, phase, zp=27.5, zpsys='ab')

        # Shape should match
        assert jax_fluxes.shape == snc_fluxes.shape == (3,)

        # Values should match
        np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4)

    def test_broadcast_matched_bands_phases(self):
        """Broadcasting: matched bands and phases (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        bands = ['bessellb', 'bessellv', 'bessellr']
        phases = [0.0, 5.0, 10.0]

        # v3.0 functional API - pass params
        jax_fluxes = jax_source.bandflux(params, bands, phases, zp=27.5, zpsys='ab')
        snc_fluxes = snc_source.bandflux(bands, phases, zp=27.5, zpsys='ab')

        # Shape should match
        assert jax_fluxes.shape == snc_fluxes.shape == (3,)

        # Values should match
        np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4)

    def test_broadcast_with_zeropoint_arrays(self):
        """Broadcasting with array zeropoints (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        bands = ['bessellb', 'bessellv', 'bessellr']
        phases = [0.0, 5.0, 10.0]
        zps = [27.5, 27.5, 27.5]
        zpsys = 'ab'

        # v3.0 functional API - pass params
        jax_fluxes = jax_source.bandflux(params, bands, phases, zp=zps, zpsys=zpsys)
        snc_fluxes = snc_source.bandflux(bands, phases, zp=zps, zpsys=zpsys)

        # Values should match
        np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4)

    def test_broadcast_with_array_zpsys(self):
        """Broadcasting with array magnitude systems (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        bands = ['bessellb', 'bessellv', 'bessellr']
        phases = [0.0, 5.0, 10.0]
        zps = [27.5, 27.5, 27.5]
        zpsys = ['ab', 'ab', 'ab']

        # v3.0 functional API - pass params
        jax_fluxes = jax_source.bandflux(params, bands, phases, zp=zps, zpsys=zpsys)
        snc_fluxes = snc_source.bandflux(bands, phases, zp=zps, zpsys=zpsys)

        # Values should match
        np.testing.assert_allclose(jax_fluxes, snc_fluxes, rtol=1e-4)


class TestZeropointHandling:
    """Test zeropoint and magnitude system handling."""

    def test_no_zeropoint(self):
        """Flux without zeropoint should be in photons/s/cm^2 (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source['x0'] = 1e-5

        # v3.0 functional API - pass params
        jax_flux = jax_source.bandflux(params, 'bessellb', 0.0)
        snc_flux = snc_source.bandflux('bessellb', 0.0)

        # Should match sncosmo
        np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4)

    def test_ab_zeropoint(self):
        """Flux with AB zeropoint (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source['x0'] = 1e-5

        # v3.0 functional API - pass params
        jax_flux = jax_source.bandflux(params, 'bessellb', 0.0, zp=27.5, zpsys='ab')
        snc_flux = snc_source.bandflux('bessellb', 0.0, zp=27.5, zpsys='ab')

        # Should match sncosmo
        np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4)

    def test_different_zeropoints(self):
        """Test different zeropoint values (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

        # Different zeropoints should give different fluxes (v3.0 functional API)
        flux_zp25 = jax_source.bandflux(params, 'bessellb', 0.0, zp=25.0, zpsys='ab')
        flux_zp27 = jax_source.bandflux(params, 'bessellb', 0.0, zp=27.5, zpsys='ab')

        # Flux ratio should match expected magnitude scaling
        # mag_diff = 2.5, so flux_ratio = 10^(0.4 * 2.5) = 10
        expected_ratio = 10**(0.4 * (27.5 - 25.0))
        actual_ratio = flux_zp27 / flux_zp25
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-4)

    def test_zp_without_zpsys_raises_error(self):
        """Providing zp without zpsys should raise ValueError (v3.0 functional API)."""
        source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

        # v3.0 functional API - pass params
        with pytest.raises(ValueError, match='zpsys'):
            source.bandflux(params, 'bessellb', 0.0, zp=27.5)


class TestBandmagMethod:
    """Test the bandmag method."""

    def test_bandmag_scalar(self):
        """Test bandmag with scalar inputs (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        # v3.0 functional API - pass params
        jax_mag = jax_source.bandmag(params, 'bessellb', 'ab', 0.0)
        snc_mag = snc_source.bandmag('bessellb', 'ab', 0.0)

        # Should match to high precision
        np.testing.assert_allclose(jax_mag, snc_mag, rtol=1e-4)

    def test_bandmag_array(self):
        """Test bandmag with array inputs (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        phases = [0.0, 5.0, 10.0]
        # v3.0 functional API - pass params
        jax_mags = jax_source.bandmag(params, 'bessellb', 'ab', phases)
        snc_mags = snc_source.bandmag('bessellb', 'ab', phases)

        # Should match to high precision
        np.testing.assert_allclose(jax_mags, snc_mags, rtol=1e-4)

    def test_bandmag_multiple_bands(self):
        """Test bandmag with multiple bands (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.5, 'c': -0.1}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source.set(x0=1e-5, x1=0.5, c=-0.1)

        bands = ['bessellb', 'bessellv', 'bessellr']
        phases = [0.0, 5.0, 10.0]
        # v3.0 functional API - pass params
        jax_mags = jax_source.bandmag(params, bands, 'ab', phases)
        snc_mags = snc_source.bandmag(bands, 'ab', phases)

        # Should match to high precision
        np.testing.assert_allclose(jax_mags, snc_mags, rtol=1e-4)


class TestPhaseWavelengthLimits:
    """Test phase and wavelength limit methods."""

    def test_minphase_maxphase(self):
        """Test minphase() and maxphase() methods."""
        jax_source = SALT3Source()
        snc_source = sncosmo.get_source('salt3-nir')

        # Should match sncosmo limits
        assert jax_source.minphase() == snc_source.minphase()
        assert jax_source.maxphase() == snc_source.maxphase()

    def test_minwave_maxwave(self):
        """Test minwave() and maxwave() methods."""
        jax_source = SALT3Source()
        snc_source = sncosmo.get_source('salt3-nir')

        # Should match sncosmo limits
        assert jax_source.minwave() == snc_source.minwave()
        assert jax_source.maxwave() == snc_source.maxwave()


class TestStringRepresentation:
    """Test string representation methods."""

    def test_str(self):
        """Test __str__ method (v3.0 - no parameter storage)."""
        source = SALT3Source()

        str_repr = str(source)
        assert 'SALT3Source' in str_repr
        # v3.0 should mention functional API
        assert 'functional' in str_repr.lower() or 'v3.0' in str_repr

    def test_repr(self):
        """Test __repr__ method."""
        source = SALT3Source()
        repr_str = repr(source)
        assert 'SALT3Source' in repr_str


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_flux_parameters(self):
        """Test with parameters that give near-zero flux (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-20, 'x1': 0.0, 'c': 0.0}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source['x0'] = 1e-20

        # v3.0 functional API - pass params
        jax_flux = jax_source.bandflux(params, 'bessellb', 0.0, zp=27.5, zpsys='ab')
        snc_flux = snc_source.bandflux('bessellb', 0.0, zp=27.5, zpsys='ab')

        # Should both be very small and match
        assert jax_flux < 1e-10
        np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4)

    def test_extreme_phase_values(self):
        """Test with phases near model boundaries (v3.0 functional API)."""
        jax_source = SALT3Source()
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

        snc_source = sncosmo.get_source('salt3-nir')
        snc_source['x0'] = 1e-5

        # Test near boundaries
        phases = [jax_source.minphase() + 1, jax_source.maxphase() - 1]

        for phase in phases:
            # v3.0 functional API - pass params
            jax_flux = jax_source.bandflux(params, 'bessellb', phase, zp=27.5, zpsys='ab')
            snc_flux = snc_source.bandflux('bessellb', phase, zp=27.5, zpsys='ab')
            np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4)

    def test_extreme_parameter_values(self):
        """Test with extreme but valid parameter values (v3.0 functional API)."""
        jax_source = SALT3Source()

        snc_source = sncosmo.get_source('salt3-nir')

        # Test extreme x1
        params1 = {'x0': 1e-5, 'x1': 2.0, 'c': 0.0}
        snc_source.set(x0=1e-5, x1=2.0, c=0.0)

        # v3.0 functional API - pass params
        jax_flux = jax_source.bandflux(params1, 'bessellb', 0.0, zp=27.5, zpsys='ab')
        snc_flux = snc_source.bandflux('bessellb', 0.0, zp=27.5, zpsys='ab')
        np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4)

        # Test extreme c
        params2 = {'x0': 1e-5, 'x1': 0.0, 'c': 0.3}
        snc_source.set(x0=1e-5, x1=0.0, c=0.3)

        # v3.0 functional API - pass params
        jax_flux = jax_source.bandflux(params2, 'bessellb', 0.0, zp=27.5, zpsys='ab')
        snc_flux = snc_source.bandflux('bessellb', 0.0, zp=27.5, zpsys='ab')
        np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-4)


def run_all_tests():
    """Run all v3.0 functional API behavior tests."""
    print("=" * 70)
    print("SALT3Source Behavior Tests (v3.0 Functional API)")
    print("Testing numerical consistency with sncosmo")
    print("=" * 70)

    # Run all test classes (v3.0 - TestParameterAccess removed)
    test_classes = [
        TestParameterNames,
        TestReturnTypes,
        TestBroadcasting,
        TestZeropointHandling,
        TestBandmagMethod,
        TestPhaseWavelengthLimits,
        TestStringRepresentation,
        TestEdgeCases,
    ]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 70)

        # Create instance and run all test methods
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                raise

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("v3.0 functional API maintains numerical consistency with sncosmo")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
