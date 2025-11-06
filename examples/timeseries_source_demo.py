"""TimeSeriesSource demonstration script.

This script shows how to use JAX-bandflux's TimeSeriesSource class to fit
custom supernova spectral energy distributions (SEDs). It demonstrates:

1. Creating a TimeSeriesSource from a 2D flux grid
2. Calculating synthetic photometry (simple mode)
3. High-performance mode with pre-computed bridges
4. Using TimeSeriesSource in JIT-compiled likelihood functions
5. Comparison with sncosmo for validation

Author: JAX-bandflux team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from jax_supernovae import TimeSeriesSource
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge
import jax
import jax.numpy as jnp

# Try to import sncosmo for comparison (optional)
try:
    import sncosmo
    HAS_SNCOSMO = True
except ImportError:
    HAS_SNCOSMO = False
    print("Note: sncosmo not available, skipping comparison plots")


# ============================================================================
# Example 1: Create a Simple Gaussian SED Model
# ============================================================================

def example_1_create_simple_model():
    """Create a simple Gaussian SED model."""
    print("=" * 70)
    print("Example 1: Creating a Simple Gaussian SED Model")
    print("=" * 70)

    # Define phase and wavelength grids
    phase = np.linspace(-20, 50, 100)  # Days
    wave = np.linspace(3000, 9000, 200)  # Angstroms

    # Create 2D grids for computation
    p_grid, w_grid = np.meshgrid(phase, wave, indexing='ij')

    # Define a Gaussian light curve in time
    time_profile = np.exp(-0.5 * (p_grid / 10.0)**2)

    # Define a Gaussian SED in wavelength (peaked at 5000 Å)
    wave_profile = np.exp(-0.5 * ((w_grid - 5000.0) / 1000.0)**2)

    # Combined flux (scaled to realistic supernova levels)
    flux = time_profile * wave_profile * 1e-15  # erg/s/cm²/Å

    # Create TimeSeriesSource
    source = TimeSeriesSource(
        phase, wave, flux,
        zero_before=False,  # Extrapolate before minphase
        time_spline_degree=3,  # Cubic interpolation (default)
        name='gaussian_sn',
        version='1.0'
    )

    print(f"\nCreated source: {source}")
    print(f"Parameters: {source.param_names}")
    print(f"Phase range: [{source.minphase():.1f}, {source.maxphase():.1f}] days")
    print(f"Wavelength range: [{source.minwave():.0f}, {source.maxwave():.0f}] Å")

    return source, phase, wave, flux


# ============================================================================
# Example 2: Calculate Synthetic Photometry (Simple Mode)
# ============================================================================

def example_2_simple_photometry(source):
    """Calculate synthetic photometry using simple mode."""
    print("\n" + "=" * 70)
    print("Example 2: Simple Synthetic Photometry")
    print("=" * 70)

    # Define parameters (functional API)
    params = {'amplitude': 1.0}

    # Calculate bandflux at single phase
    print("\nSingle observation:")
    phase_single = 0.0
    band_single = 'bessellb'
    flux_single = source.bandflux(params, band_single, phase_single,
                                  zp=25.0, zpsys='ab')
    print(f"  Band: {band_single}, Phase: {phase_single:.1f} days")
    print(f"  Flux: {flux_single:.6e} (with zp=25.0)")

    # Calculate bandflux at multiple phases (light curve)
    print("\nLight curve (B-band):")
    phases = np.linspace(-10, 30, 20)
    fluxes_b = source.bandflux(params, 'bessellb', phases, zp=25.0, zpsys='ab')

    print(f"  Phases: {phases[0]:.1f} to {phases[-1]:.1f} days ({len(phases)} points)")
    print(f"  Flux range: {fluxes_b.min():.6e} to {fluxes_b.max():.6e}")

    # Calculate multi-band observation
    print("\nMulti-band snapshot at phase=0:")
    bands_multi = ['bessellb', 'bessellv', 'bessellr', 'besselli']
    phases_multi = np.zeros(len(bands_multi))  # All at phase=0
    fluxes_multi = source.bandflux(params, bands_multi, phases_multi,
                                   zp=25.0, zpsys='ab')

    for band, flux in zip(bands_multi, fluxes_multi):
        mag = -2.5 * np.log10(flux)  # Convert to magnitude
        print(f"  {band:>10}: flux={flux:.6e}, mag={mag:.3f}")

    return phases, fluxes_b


# ============================================================================
# Example 3: High-Performance Mode (Pre-computed Bridges)
# ============================================================================

def example_3_optimised_mode(source):
    """Demonstrate high-performance mode with pre-computed bridges."""
    print("\n" + "=" * 70)
    print("Example 3: High-Performance Mode (Optimised)")
    print("=" * 70)

    # Simulate a realistic light curve: 30 observations in 3 bands
    n_obs = 30
    phases = np.linspace(-10, 40, n_obs)

    # Observations cycle through B, V, R bands
    band_names = ['bessellb', 'bessellv', 'bessellr'] * (n_obs // 3)
    band_names += ['bessellb'] * (n_obs % 3)  # Fill remainder

    zps = np.ones(n_obs) * 25.0

    print(f"\nSimulated light curve: {n_obs} observations")
    print(f"Bands: {set(band_names)}")
    print(f"Phase range: {phases.min():.1f} to {phases.max():.1f} days")

    # -------------------------
    # Method 1: Simple mode (slower)
    # -------------------------
    print("\nMethod 1: Simple mode (creates bridges on-the-fly)")
    import time
    params = {'amplitude': 1.0}

    start = time.time()
    fluxes_simple = source.bandflux(params, band_names, phases,
                                    zp=zps, zpsys='ab')
    time_simple = time.time() - start
    print(f"  Time: {time_simple*1000:.2f} ms")
    print(f"  Flux range: {fluxes_simple.min():.6e} to {fluxes_simple.max():.6e}")

    # -------------------------
    # Method 2: Optimised mode (faster)
    # -------------------------
    print("\nMethod 2: Optimised mode (pre-computed bridges)")

    # Pre-compute bridges ONCE for unique bands
    unique_bands = ['bessellb', 'bessellv', 'bessellr']
    bridges = tuple(precompute_bandflux_bridge(get_bandpass(b))
                   for b in unique_bands)

    # Create band indices mapping each observation to its bridge
    band_to_idx = {b: i for i, b in enumerate(unique_bands)}
    band_indices = jnp.array([band_to_idx[b] for b in band_names])

    start = time.time()
    fluxes_optimised = source.bandflux(params, None, phases,
                                       zp=zps, zpsys='ab',
                                       band_indices=band_indices,
                                       bridges=bridges,
                                       unique_bands=unique_bands)
    time_optimised = time.time() - start
    print(f"  Time: {time_optimised*1000:.2f} ms")
    print(f"  Flux range: {fluxes_optimised.min():.6e} to {fluxes_optimised.max():.6e}")

    # Verify results match
    max_diff = np.max(np.abs(fluxes_simple - fluxes_optimised))
    print(f"\nMaximum difference: {max_diff:.2e} (should be ~0)")
    print(f"Speedup: {time_simple/time_optimised:.1f}x faster")

    return phases, band_names, fluxes_optimised, bridges, band_indices, unique_bands


# ============================================================================
# Example 4: JIT-Compiled Likelihood Function
# ============================================================================

def example_4_jit_likelihood(source, phases, band_names, bridges,
                            band_indices, unique_bands):
    """Use TimeSeriesSource in JIT-compiled likelihood function."""
    print("\n" + "=" * 70)
    print("Example 4: JIT-Compiled Likelihood Function")
    print("=" * 70)

    # Simulate observed data with noise
    np.random.seed(42)
    true_amplitude = 1.0
    params_true = {'amplitude': true_amplitude}

    # Generate "observed" fluxes with noise
    phases_jnp = jnp.array(phases)
    zps = jnp.ones(len(phases)) * 25.0

    true_fluxes = source.bandflux(params_true, None, phases_jnp,
                                  zp=zps, zpsys='ab',
                                  band_indices=band_indices,
                                  bridges=bridges,
                                  unique_bands=unique_bands)

    # Add 5% noise
    noise_level = 0.05
    flux_errors = true_fluxes * noise_level
    observed_fluxes = true_fluxes + jnp.array(np.random.normal(0, 1, len(phases))) * flux_errors

    print(f"\nSimulated observations:")
    print(f"  True amplitude: {true_amplitude}")
    print(f"  Number of observations: {len(phases)}")
    print(f"  Noise level: {noise_level*100:.1f}%")

    # Define JIT-compiled log-likelihood function
    @jax.jit
    def loglikelihood(amplitude):
        """Calculate log-likelihood for given amplitude."""
        params = {'amplitude': amplitude}

        # Calculate model fluxes
        model_fluxes = source.bandflux(params, None, phases_jnp,
                                       zp=zps, zpsys='ab',
                                       band_indices=band_indices,
                                       bridges=bridges,
                                       unique_bands=unique_bands)

        # Chi-squared
        chi2 = jnp.sum(((observed_fluxes - model_fluxes) / flux_errors)**2)

        # Log-likelihood (Gaussian)
        return -0.5 * chi2

    # Test the likelihood at different amplitudes
    print("\nTesting likelihood function:")
    test_amplitudes = jnp.array([0.8, 0.9, 1.0, 1.1, 1.2])

    # First call compiles the function
    import time
    start = time.time()
    logL_first = loglikelihood(test_amplitudes[0])
    time_first = time.time() - start

    # Subsequent calls are much faster (already compiled)
    start = time.time()
    logL_values = jax.vmap(loglikelihood)(test_amplitudes)
    time_compiled = time.time() - start

    print(f"\n  {'Amplitude':>12} {'Log-Likelihood':>15}")
    print("  " + "-" * 30)
    for amp, logL in zip(test_amplitudes, logL_values):
        print(f"  {amp:12.2f} {logL:15.2f}")

    best_idx = jnp.argmax(logL_values)
    print(f"\n  Best-fit amplitude: {test_amplitudes[best_idx]:.2f}")
    print(f"  True amplitude: {true_amplitude:.2f}")

    print(f"\nPerformance:")
    print(f"  First call (with compilation): {time_first*1000:.2f} ms")
    print(f"  Compiled calls (5 amplitudes): {time_compiled*1000:.2f} ms")
    print(f"  Per evaluation: {time_compiled/len(test_amplitudes)*1000:.3f} ms")

    return logL_values


# ============================================================================
# Example 5: Comparison with sncosmo
# ============================================================================

def example_5_sncosmo_comparison(source, phase, wave, flux):
    """Compare JAX-bandflux with sncosmo."""
    if not HAS_SNCOSMO:
        print("\n" + "=" * 70)
        print("Example 5: Comparison with sncosmo (SKIPPED - sncosmo not available)")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("Example 5: Comparison with sncosmo")
    print("=" * 70)

    # Create sncosmo source
    snc_source = sncosmo.TimeSeriesSource(phase, wave, flux)
    snc_source.set(amplitude=1.0)

    # Parameters
    params = {'amplitude': 1.0}

    # Test phases
    test_phases = np.array([-10, 0, 10, 20, 30])
    test_bands = ['bessellb', 'bessellv', 'bessellr']

    print("\nComparing bandflux across different bands:")
    print(f"\n{'Band':>10} {'Phase':>8} {'JAX Flux':>15} {'SNCosmo Flux':>15} {'Rel. Error':>12}")
    print("-" * 75)

    all_errors = []
    for band in test_bands:
        for phase_val in test_phases:
            jax_flux = source.bandflux(params, band, phase_val, zp=25.0, zpsys='ab')
            snc_flux = snc_source.bandflux(band, phase_val, zp=25.0, zpsys='ab')

            rel_error = abs(jax_flux - snc_flux) / abs(snc_flux) if abs(snc_flux) > 0 else 0
            all_errors.append(rel_error)

            print(f"{band:>10} {phase_val:8.0f} {jax_flux:15.6e} {snc_flux:15.6e} {rel_error:12.2e}")

    max_error = max(all_errors)
    mean_error = np.mean(all_errors)

    print(f"\nStatistics:")
    print(f"  Maximum relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    print(f"  Target tolerance: 1e-4 (0.01%)")

    if max_error < 1e-4:
        print(f"  ✓ PASSED: All errors below tolerance")
    else:
        print(f"  ✗ FAILED: Some errors exceed tolerance")


# ============================================================================
# Example 6: Plotting Light Curves
# ============================================================================

def example_6_plot_light_curves(source, phases_lc, fluxes_lc):
    """Plot light curves in multiple bands."""
    print("\n" + "=" * 70)
    print("Example 6: Plotting Light Curves")
    print("=" * 70)

    params = {'amplitude': 1.0}

    # Calculate light curves in multiple bands
    bands_plot = ['bessellb', 'bessellv', 'bessellr', 'besselli']
    colors = ['blue', 'green', 'red', 'maroon']
    phases_dense = np.linspace(-15, 45, 100)

    plt.figure(figsize=(10, 6))

    for band, color in zip(bands_plot, colors):
        fluxes = source.bandflux(params, band, phases_dense, zp=25.0, zpsys='ab')
        mags = -2.5 * np.log10(np.array(fluxes))

        plt.plot(phases_dense, mags, label=band, color=color, linewidth=2)

    plt.xlabel('Phase (days)', fontsize=12)
    plt.ylabel('AB Magnitude', fontsize=12)
    plt.title('Gaussian SED Light Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.gca().invert_yaxis()  # Magnitudes increase downward
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = 'timeseries_light_curves.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Saved light curves to: {output_file}")
    plt.close()


# ============================================================================
# Main Demonstration
# ============================================================================

def main():
    """Run all demonstration examples."""
    print("\n" + "#" * 70)
    print("# JAX-bandflux TimeSeriesSource Demonstration")
    print("#" * 70)

    # Example 1: Create model
    source, phase, wave, flux = example_1_create_simple_model()

    # Example 2: Simple photometry
    phases_lc, fluxes_lc = example_2_simple_photometry(source)

    # Example 3: Optimised mode
    phases_opt, band_names, fluxes_opt, bridges, band_indices, unique_bands = \
        example_3_optimised_mode(source)

    # Example 4: JIT-compiled likelihood
    logL_values = example_4_jit_likelihood(source, phases_opt, band_names,
                                          bridges, band_indices, unique_bands)

    # Example 5: Compare with sncosmo
    example_5_sncosmo_comparison(source, phase, wave, flux)

    # Example 6: Plot light curves
    try:
        example_6_plot_light_curves(source, phases_lc, fluxes_lc)
    except Exception as e:
        print(f"\nNote: Plotting skipped ({e})")

    print("\n" + "#" * 70)
    print("# Demonstration Complete!")
    print("#" * 70)
    print("\nKey Takeaways:")
    print("  1. TimeSeriesSource enables fitting custom SED models")
    print("  2. Functional API: params passed as dict to methods")
    print("  3. Two modes: simple (convenient) vs optimised (fast)")
    print("  4. JIT-compiled for high performance in MCMC/nested sampling")
    print("  5. Matches sncosmo numerically to <0.01%")
    print("\nFor more information, see the documentation:")
    print("  - docs/timeseries_source.md")
    print("  - tests/test_timeseries_source.py")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
