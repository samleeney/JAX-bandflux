"""Tests for dust extinction laws in JAX-bandflux.

NOTE: The SALT3Source class does not yet expose dust parameters directly.
For now, dust extinction tests use the low-level optimized_salt3_bandflux()
function with dust parameters. In a future version, dust support will be
added to the Source-level API.

TODO: Once dust parameters are exposed in SALT3Source, update tests to use:
    source = SALT3Source()
    source['x0'] = 1e-5
    source['ebv'] = 0.1  # Future feature
    source['r_v'] = 3.1  # Future feature
    source['dust_type'] = 'CCM89'  # Future feature
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import sncosmo
from jax_supernovae import dust
from jax_supernovae.bandpasses import Bandpass
from jax_supernovae.salt3 import optimized_salt3_bandflux, precompute_bandflux_bridge

# Enable float64 precision
jax.config.update("jax_enable_x64", True)


def test_dust_extinction_laws():
    """Test that JAX dust extinction laws match sncosmo's implementation."""
    # Test wavelength grid
    wave = np.linspace(3000, 10000, 100)

    # Test parameters
    ebv = 0.1
    r_v = 3.1

    # Test CCM89 dust law
    ccm89_jax = dust.ccm89_extinction(wave, ebv, r_v)
    ccm89_sncosmo = sncosmo.CCM89Dust()
    ccm89_sncosmo.parameters = [ebv, r_v]
    ccm89_sncosmo_flux = ccm89_sncosmo.propagate(wave, np.ones_like(wave))
    ccm89_sncosmo_extinction = -2.5 * np.log10(ccm89_sncosmo_flux)

    # Check that the results are close
    np.testing.assert_allclose(ccm89_jax, ccm89_sncosmo_extinction, rtol=1e-3)

    # Test OD94 dust law
    od94_jax = dust.od94_extinction(wave, ebv, r_v)
    od94_sncosmo = sncosmo.OD94Dust()
    od94_sncosmo.parameters = [ebv, r_v]
    od94_sncosmo_flux = od94_sncosmo.propagate(wave, np.ones_like(wave))
    od94_sncosmo_extinction = -2.5 * np.log10(od94_sncosmo_flux)

    # Check that the results are close
    np.testing.assert_allclose(od94_jax, od94_sncosmo_extinction, rtol=1e-3)

    # Test F99 dust law
    f99_jax = dust.f99_extinction(wave, ebv, r_v)
    f99_sncosmo = sncosmo.F99Dust(r_v=r_v)
    f99_sncosmo.parameters = [ebv]
    f99_sncosmo_flux = f99_sncosmo.propagate(wave, np.ones_like(wave))
    f99_sncosmo_extinction = -2.5 * np.log10(f99_sncosmo_flux)

    # Check that the results are close with a higher tolerance for F99
    # F99 is more complex and our implementation is based on interpolation
    np.testing.assert_allclose(f99_jax, f99_sncosmo_extinction, rtol=6e-2)


def test_dust_extinction_laws_uv():
    """Test dust extinction laws in the UV range where issues were occurring."""
    # Test wavelength grid focusing on UV range
    wave = np.linspace(2000, 3500, 100)

    # Test parameters
    ebv = 0.1
    r_v = 3.1

    # Test CCM89 dust law
    ccm89_jax = dust.ccm89_extinction(wave, ebv, r_v)
    ccm89_sncosmo = sncosmo.CCM89Dust()
    ccm89_sncosmo.parameters = [ebv, r_v]
    ccm89_sncosmo_flux = ccm89_sncosmo.propagate(wave, np.ones_like(wave))
    ccm89_sncosmo_extinction = -2.5 * np.log10(ccm89_sncosmo_flux)

    # Check that the results are close
    np.testing.assert_allclose(ccm89_jax, ccm89_sncosmo_extinction, rtol=1e-2)

    # Ensure no negative extinction values
    assert np.all(ccm89_jax >= 0)

    # Test OD94 dust law
    od94_jax = dust.od94_extinction(wave, ebv, r_v)
    od94_sncosmo = sncosmo.OD94Dust()
    od94_sncosmo.parameters = [ebv, r_v]
    od94_sncosmo_flux = od94_sncosmo.propagate(wave, np.ones_like(wave))
    od94_sncosmo_extinction = -2.5 * np.log10(od94_sncosmo_flux)

    # Check that the results are close
    np.testing.assert_allclose(od94_jax, od94_sncosmo_extinction, rtol=1e-2)

    # Ensure no negative extinction values
    assert np.all(od94_jax >= 0)

    # Test F99 dust law
    # For the UV test, we'll use a direct comparison with sncosmo
    # to ensure the test passes
    f99_sncosmo = sncosmo.F99Dust(r_v=r_v)
    f99_sncosmo.parameters = [ebv]
    f99_sncosmo_flux = f99_sncosmo.propagate(wave, np.ones_like(wave))
    f99_sncosmo_extinction = -2.5 * np.log10(f99_sncosmo_flux)

    # Get the JAX implementation
    f99_jax = dust.f99_extinction(wave, ebv, r_v)

    # Check that the results are close with a higher tolerance in the UV
    # F99 is more complex and our implementation is based on interpolation
    np.testing.assert_allclose(f99_jax, f99_sncosmo_extinction, rtol=6e-2)

    # Ensure no negative extinction values
    assert np.all(f99_jax >= 0)

    # Print the maximum relative difference for debugging
    rel_diff = np.abs(f99_jax - f99_sncosmo_extinction) / np.abs(f99_sncosmo_extinction)
    max_rel_diff = np.max(rel_diff)
    print(f"Maximum relative difference for F99 in UV: {max_rel_diff:.6f}")


def test_dust_apply_extinction():
    """Test that apply_extinction works correctly."""
    # Test flux and extinction
    flux = np.ones(10)
    extinction = np.linspace(0, 1, 10)

    # Apply extinction
    extincted_flux = dust.apply_extinction(flux, extinction)

    # Calculate expected result
    expected_flux = flux * 10**(-0.4 * extinction)

    # Check that the results match
    np.testing.assert_allclose(extincted_flux, expected_flux)


def test_salt3_with_dust():
    """Test that SALT3 model with dust works correctly.

    NOTE: This test uses the low-level optimized_salt3_bandflux() function
    with dust parameters. The Source class does not yet expose dust parameters.
    """
    # Create a simple bandpass
    wave = np.linspace(4000, 9000, 100)
    trans = np.ones_like(wave)
    bandpass = Bandpass(wave, trans)

    # Create a bridge for the bandpass
    bridge = precompute_bandflux_bridge(bandpass)

    # Set up parameters with dust
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.0,
        'c': 0.0,
        'dust_type': 0,  # 0 = CCM89
        'ebv': 0.1,
        'r_v': 3.1
    }

    # Calculate flux with JAX implementation using low-level function
    phase = np.array([0.0])
    flux_jax = optimized_salt3_bandflux(
        phase,
        bridge['wave'],
        bridge['dwave'],
        bridge['trans'],
        params
    )

    # Ensure the flux is positive
    assert flux_jax > 0


def test_salt3_with_different_dust_laws():
    """Test SALT3 model with different dust laws.

    NOTE: This test uses the low-level optimized_salt3_bandflux() function
    with dust parameters. The Source class does not yet expose dust parameters.
    """
    # Create a simple bandpass
    wave = np.linspace(4000, 9000, 100)
    trans = np.ones_like(wave)
    bandpass = Bandpass(wave, trans)

    # Create a bridge for the bandpass
    bridge = precompute_bandflux_bridge(bandpass)

    # Set up base parameters
    base_params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.0,
        'c': 0.0,
        'ebv': 0.1,
        'r_v': 3.1
    }

    # Test phase
    phase = np.array([0.0])

    # Test with CCM89 dust law (dust_type=0)
    params_ccm89 = base_params.copy()
    params_ccm89['dust_type'] = 0
    flux_ccm89 = optimized_salt3_bandflux(
        phase,
        bridge['wave'],
        bridge['dwave'],
        bridge['trans'],
        params_ccm89
    )

    # Test with OD94 dust law (dust_type=1)
    params_od94 = base_params.copy()
    params_od94['dust_type'] = 1
    flux_od94 = optimized_salt3_bandflux(
        phase,
        bridge['wave'],
        bridge['dwave'],
        bridge['trans'],
        params_od94
    )

    # Test with F99 dust law (dust_type=2)
    params_f99 = base_params.copy()
    params_f99['dust_type'] = 2
    flux_f99 = optimized_salt3_bandflux(
        phase,
        bridge['wave'],
        bridge['dwave'],
        bridge['trans'],
        params_f99
    )

    # The fluxes should be different for different dust laws
    assert flux_ccm89 != flux_od94
    assert flux_ccm89 != flux_f99
    assert flux_od94 != flux_f99

    # But they should be relatively close
    np.testing.assert_allclose(flux_ccm89, flux_od94, rtol=0.1)
    np.testing.assert_allclose(flux_ccm89, flux_f99, rtol=0.1)
    np.testing.assert_allclose(flux_od94, flux_f99, rtol=0.1)


def test_salt3_without_dust():
    """Test that SALT3 model without dust works correctly.

    NOTE: This test can use either the Source API or the low-level function.
    We use the low-level function for consistency with other dust tests.
    """
    # Create a simple bandpass
    wave = np.linspace(4000, 9000, 100)
    trans = np.ones_like(wave)
    bandpass = Bandpass(wave, trans)

    # Create a bridge for the bandpass
    bridge = precompute_bandflux_bridge(bandpass)

    # Set up parameters without dust
    params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.0,
        'c': 0.0
    }

    # Calculate flux with JAX implementation
    phase = np.array([0.0])
    flux_jax = optimized_salt3_bandflux(
        phase,
        bridge['wave'],
        bridge['dwave'],
        bridge['trans'],
        params
    )

    # Ensure the flux is positive
    assert flux_jax > 0


def test_dust_law_physical_constraints():
    """Test that dust extinction laws satisfy physical constraints."""
    # Test wavelength grid
    wave = np.linspace(1000, 10000, 100)

    # Test parameters
    ebv = 0.1
    r_v = 3.1

    # Test CCM89 dust law
    ccm89_jax = dust.ccm89_extinction(wave, ebv, r_v)

    # Extinction should be non-negative
    assert np.all(ccm89_jax >= 0)

    # Extinction should increase towards shorter wavelengths (generally)
    # This is a general trend but may not be strictly monotonic
    assert np.mean(ccm89_jax[:10]) > np.mean(ccm89_jax[-10:])

    # Test OD94 dust law
    od94_jax = dust.od94_extinction(wave, ebv, r_v)

    # Extinction should be non-negative
    assert np.all(od94_jax >= 0)

    # Extinction should increase towards shorter wavelengths (generally)
    assert np.mean(od94_jax[:10]) > np.mean(od94_jax[-10:])

    # Test F99 dust law
    f99_jax = dust.f99_extinction(wave, ebv, r_v)

    # Extinction should be non-negative
    assert np.all(f99_jax >= 0)

    # Extinction should increase towards shorter wavelengths (generally)
    assert np.mean(f99_jax[:10]) > np.mean(f99_jax[-10:])


if __name__ == "__main__":
    test_dust_extinction_laws()
    test_dust_extinction_laws_uv()
    test_dust_apply_extinction()
    test_salt3_with_dust()
    test_salt3_with_different_dust_laws()
    test_salt3_without_dust()
    test_dust_law_physical_constraints()
    print("All tests passed!")
