"""Test transmission shift functionality for bandpasses.

This module tests wavelength-dependent transmission shifts in bandpass filters.
These shifts can be used to model systematic effects or instrumental variations.

NOTE: Transmission shifts are not yet exposed in the SALT3Source API.
These tests verify the underlying optimized functions still work correctly.
TODO: Add transmission shift support to SALT3Source.bandflux() in future release.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax_supernovae.bandpasses import Bandpass
from jax_supernovae.salt3 import optimized_salt3_bandflux, precompute_bandflux_bridge


def test_bandpass_constant_shift():
    """Test that constant wavelength shift works at bandpass level.

    A constant shift should move the transmission peak by approximately
    the shift amount. This is the fundamental test for shift functionality.
    """
    # Create test bandpass with Gaussian transmission
    wave = jnp.linspace(4000, 7000, 1000)
    trans = jnp.exp(-(wave - 5500)**2 / (2 * 500**2))
    bp = Bandpass(wave, trans)

    # Apply shift
    shift = 50.0  # 50 Angstrom shift
    trans_original = bp(wave)
    trans_shifted = bp(wave, shift=shift)

    # Find peaks
    peak_original = wave[jnp.argmax(trans_original)]
    peak_shifted = wave[jnp.argmax(trans_shifted)]

    # Peak should move by approximately +shift
    # (Positive shift means effective wavelength is wave - shift,
    # so transmission peak appears to move right)
    peak_diff = peak_shifted - peak_original
    assert jnp.abs(peak_diff - shift) < 5.0, \
        f"Peak moved by {peak_diff:.1f} Å, expected ~{shift:.1f} Å"

    print(f"\n✓ Constant shift test passed:")
    print(f"  Shift applied: {shift:.1f} Å")
    print(f"  Peak moved: {peak_diff:.1f} Å")


def test_flux_with_transmission_shift():
    """Test flux calculation with transmission shift.

    This tests the underlying optimized_salt3_bandflux function with shift
    parameters. The shifted and unshifted fluxes should be detectably different.
    """
    # Create narrower Gaussian bandpass for sensitivity to shifts
    wave = jnp.linspace(4000, 7000, 300)
    trans = jnp.exp(-(wave - 5500)**2 / (2 * 200**2))
    bp = Bandpass(wave, trans)

    # Prepare bridge with shift support
    bridge = precompute_bandflux_bridge(bp)

    # Model parameters
    params = {'z': 0.1, 't0': 0.0, 'x0': 1e-10, 'x1': 0.0, 'c': 0.0}
    phase = jnp.array([0.0])

    # Calculate flux without shift
    flux_original = optimized_salt3_bandflux(
        phase, bridge['wave'], bridge['dwave'], bridge['trans'],
        params
    )

    # Calculate flux with large shift for noticeable effect
    flux_shifted = optimized_salt3_bandflux(
        phase, bridge['wave'], bridge['dwave'], bridge['trans'],
        params,
        shift=500.0,  # Large shift
        wave_original=bridge['wave_original'],
        trans_original=bridge['trans_original']
    )

    # Fluxes should be significantly different
    relative_diff = jnp.abs(flux_original - flux_shifted) / jnp.abs(flux_original)
    assert relative_diff[0] > 0.01, \
        f"Relative difference {relative_diff[0]:.3%} too small, shift may not be working"

    print(f"\n✓ Flux with shift test passed:")
    print(f"  Original flux: {float(flux_original[0]):.6e}")
    print(f"  Shifted flux:  {float(flux_shifted[0]):.6e}")
    print(f"  Relative diff: {float(relative_diff[0]):.2%}")


def test_shift_gradient():
    """Test that gradients flow through shift parameter.

    This is important for optimization and MCMC applications where
    transmission shifts might be fit parameters.
    """
    import jax

    # Setup with narrow bandpass
    wave = jnp.linspace(4000, 7000, 300)
    trans = jnp.exp(-(wave - 5500)**2 / (2 * 150**2))
    bp = Bandpass(wave, trans)
    bridge = precompute_bandflux_bridge(bp)

    # Use higher flux amplitude for meaningful gradients
    params = {'z': 0.01, 't0': 0.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
    phase = jnp.array([0.0])

    # Define loss function with shift as parameter
    def loss_fn(shift):
        flux = optimized_salt3_bandflux(
            phase, bridge['wave'], bridge['dwave'], bridge['trans'],
            params,
            shift=shift,
            wave_original=bridge['wave_original'],
            trans_original=bridge['trans_original']
        )
        return jnp.sum(flux**2)

    # Compute gradient at shift=0
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(0.0)

    # Gradient should be finite and reasonable
    assert jnp.isfinite(grad), "Gradient should be finite"
    assert jnp.abs(grad) < 1e-5, "Gradient magnitude should be reasonable"

    # Verify gradient changes with shift position
    grad2 = grad_fn(100.0)
    assert jnp.isfinite(grad2), "Gradient at shift=100 should be finite"

    # Check finite differences match gradient
    eps = 1.0
    loss_plus = loss_fn(eps)
    loss_minus = loss_fn(-eps)
    finite_diff = (loss_plus - loss_minus) / (2 * eps)

    # Gradient should approximately match finite difference
    grad_at_zero = grad_fn(0.0)
    relative_error = jnp.abs(finite_diff - grad_at_zero) / (jnp.abs(finite_diff) + 1e-10)
    assert relative_error < 0.12, \
        f"Gradient error {relative_error:.2%} too large"

    print(f"\n✓ Gradient test passed:")
    print(f"  Gradient at shift=0:   {float(grad_at_zero):.6e}")
    print(f"  Finite difference:     {float(finite_diff):.6e}")
    print(f"  Relative error:        {float(relative_error):.2%}")


def test_shift_sign_convention():
    """Test the sign convention for transmission shifts.

    Positive shift should make effective wavelength = wave - shift,
    causing the transmission to appear shifted to longer wavelengths.
    """
    # Create asymmetric bandpass for clear directionality
    wave = jnp.linspace(4000, 6000, 500)
    # Asymmetric profile with steep blue edge, gradual red edge
    trans = jnp.exp(-(wave - 5000)**2 / (2 * 300**2)) * (1.0 + 0.3 * (wave - 5000) / 300)
    trans = jnp.clip(trans, 0, 1)
    bp = Bandpass(wave, trans)

    # Test positive shift
    positive_shift = 100.0
    trans_shifted_pos = bp(wave, shift=positive_shift)

    # Test negative shift
    negative_shift = -100.0
    trans_shifted_neg = bp(wave, shift=negative_shift)

    # At a wavelength on the blue side (4800 Å):
    # - Positive shift samples from wave - shift (even bluer, e.g., 4700 Å)
    # - On asymmetric profile, going more blue decreases transmission (multiplier decreases)
    # - Negative shift samples from wave + shift (redder, e.g., 4900 Å)
    # - Going more red increases transmission (multiplier increases)
    test_wave_idx = jnp.argmin(jnp.abs(wave - 4800.0))
    trans_orig = bp(wave)[test_wave_idx]
    trans_pos = trans_shifted_pos[test_wave_idx]
    trans_neg = trans_shifted_neg[test_wave_idx]

    # For positive shift, we sample from wave - shift (more blue), so expect lower transmission
    assert trans_pos < trans_orig, "Positive shift should decrease transmission on blue side"
    # For negative shift, we sample from wave + shift (more red), so expect higher transmission
    assert trans_neg > trans_orig, "Negative shift should increase transmission on blue side"

    print(f"\n✓ Sign convention test passed:")
    print(f"  Original transmission at 4800 Å: {float(trans_orig):.4f}")
    print(f"  With +100 Å shift:               {float(trans_pos):.4f}")
    print(f"  With -100 Å shift:               {float(trans_neg):.4f}")


def test_zero_shift_identity():
    """Test that zero shift gives identical results to no shift.

    This is a sanity check that the shift=0 case is handled correctly.
    """
    # Create bandpass
    wave = jnp.linspace(4000, 7000, 300)
    trans = jnp.exp(-(wave - 5500)**2 / (2 * 300**2))
    bp = Bandpass(wave, trans)
    bridge = precompute_bandflux_bridge(bp)

    # Model parameters
    params = {'z': 0.1, 't0': 0.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
    phase = jnp.array([0.0])

    # Calculate flux without shift parameter
    flux_no_shift = optimized_salt3_bandflux(
        phase, bridge['wave'], bridge['dwave'], bridge['trans'],
        params
    )

    # Calculate flux with shift=0
    flux_zero_shift = optimized_salt3_bandflux(
        phase, bridge['wave'], bridge['dwave'], bridge['trans'],
        params,
        shift=0.0,
        wave_original=bridge['wave_original'],
        trans_original=bridge['trans_original']
    )

    # Should be identical
    np.testing.assert_allclose(flux_no_shift, flux_zero_shift, rtol=1e-10,
                              err_msg="Zero shift should give identical results")

    print(f"\n✓ Zero shift identity test passed:")
    print(f"  Flux without shift: {float(flux_no_shift[0]):.10e}")
    print(f"  Flux with shift=0:  {float(flux_zero_shift[0]):.10e}")


def test_shift_with_multiple_phases():
    """Test that shifts work correctly with multiple phase observations.

    This ensures shift functionality is compatible with vectorized calculations.
    """
    # Create bandpass
    wave = jnp.linspace(4000, 7000, 300)
    trans = jnp.exp(-(wave - 5500)**2 / (2 * 250**2))
    bp = Bandpass(wave, trans)
    bridge = precompute_bandflux_bridge(bp)

    # Model parameters
    params = {'z': 0.1, 't0': 0.0, 'x0': 1e-5, 'x1': 0.5, 'c': -0.1}
    phases = jnp.array([-10.0, 0.0, 10.0, 20.0])

    # Calculate fluxes without shift
    fluxes_no_shift = optimized_salt3_bandflux(
        phases, bridge['wave'], bridge['dwave'], bridge['trans'],
        params
    )

    # Calculate fluxes with shift
    fluxes_with_shift = optimized_salt3_bandflux(
        phases, bridge['wave'], bridge['dwave'], bridge['trans'],
        params,
        shift=200.0,
        wave_original=bridge['wave_original'],
        trans_original=bridge['trans_original']
    )

    # All fluxes should be different
    assert fluxes_no_shift.shape == fluxes_with_shift.shape, "Shape mismatch"
    assert fluxes_no_shift.shape == (4,), "Expected 4 flux values"

    # Check that fluxes changed
    relative_diffs = jnp.abs(fluxes_no_shift - fluxes_with_shift) / jnp.abs(fluxes_no_shift)
    assert jnp.all(relative_diffs > 0.001), "Shift should affect all phase values"

    print(f"\n✓ Multiple phases with shift test passed:")
    print(f"  {'Phase':>8} {'Original Flux':>15} {'Shifted Flux':>15} {'Rel. Diff':>10}")
    print("  " + "-" * 54)
    for phase, flux_orig, flux_shift, rel_diff in zip(phases, fluxes_no_shift,
                                                        fluxes_with_shift, relative_diffs):
        print(f"  {float(phase):8.1f} {float(flux_orig):15.6e} {float(flux_shift):15.6e} "
              f"{float(rel_diff):10.3%}")


def run_all_tests():
    """Run all transmission shift tests."""
    print("=" * 70)
    print("Transmission Shift Tests")
    print("Testing bandpass transmission shift functionality")
    print("=" * 70)
    print("\nNOTE: Transmission shifts are not yet in SALT3Source API")
    print("These tests verify the underlying optimized functions work correctly")
    print("TODO: Add shift parameter to SALT3Source.bandflux() in future release")
    print("=" * 70)

    test_bandpass_constant_shift()
    test_flux_with_transmission_shift()
    test_shift_gradient()
    test_shift_sign_convention()
    test_zero_shift_identity()
    test_shift_with_multiple_phases()

    print("\n" + "=" * 70)
    print("ALL TRANSMISSION SHIFT TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    import os
    import sys
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

    run_all_tests()
