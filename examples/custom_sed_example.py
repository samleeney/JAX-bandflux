"""Custom SED Example: Loading and Fitting Real Spectral Energy Distributions

This script demonstrates how to create TimeSeriesSource models from real SED data:

1. Using sncosmo's built-in spectral templates (Hsiao Type Ia template)
2. Loading SED data from text files
3. Creating custom SEDs from analytical models
4. Fitting custom SEDs to photometric observations
5. Comparing custom SEDs with SALT3

Author: JAX-bandflux team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from jax_supernovae import TimeSeriesSource, SALT3Source
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge
import jax
import jax.numpy as jnp

# Try to import sncosmo for spectral templates
try:
    import sncosmo
    HAS_SNCOSMO = True
except ImportError:
    HAS_SNCOSMO = False
    print("Warning: sncosmo not available. Some examples will be skipped.")


# ============================================================================
# Example 1: Load Hsiao Type Ia Template from sncosmo
# ============================================================================

def example_1_hsiao_template():
    """Create TimeSeriesSource from sncosmo's Hsiao Type Ia template."""
    print("=" * 70)
    print("Example 1: Loading Hsiao Type Ia Template")
    print("=" * 70)

    if not HAS_SNCOSMO:
        print("Skipping - sncosmo not available\n")
        return None

    # Load Hsiao template from sncosmo
    print("\nLoading Hsiao template from sncosmo...")
    hsiao = sncosmo.get_source('hsiao')

    # Define phase and wavelength grids
    # Hsiao template is defined from -20 to +85 days
    phases = np.arange(-20.0, 86.0, 1.0)  # 1-day spacing
    wavelengths = np.arange(2000.0, 10000.0, 10.0)  # 10 Angstrom spacing

    print(f"Phase range: [{phases[0]:.1f}, {phases[-1]:.1f}] days ({len(phases)} points)")
    print(f"Wavelength range: [{wavelengths[0]:.0f}, {wavelengths[-1]:.0f} Å ({len(wavelengths)} points)")

    # Create 2D flux grid by evaluating Hsiao at all phase/wavelength points
    print("\nGenerating 2D flux grid...")
    flux_grid = np.zeros((len(phases), len(wavelengths)))

    for i, phase in enumerate(phases):
        # Get flux spectrum at this phase
        flux_grid[i, :] = hsiao._flux(phase, wavelengths)

    print(f"Flux grid shape: {flux_grid.shape}")
    print(f"Flux range: [{np.min(flux_grid):.3e}, {np.max(flux_grid):.3e}] erg/s/cm²/Å")

    # Create TimeSeriesSource
    print("\nCreating TimeSeriesSource from Hsiao template...")
    source_jax = TimeSeriesSource(
        phases, wavelengths, flux_grid,
        zero_before=True,
        time_spline_degree=3,
        name='hsiao',
        version='sncosmo'
    )

    print(f"✓ Created JAX TimeSeriesSource")
    print(f"  Parameters: {source_jax.param_names}")
    print(f"  Phase range: [{source_jax.minphase():.1f}, {source_jax.maxphase():.1f}] days")
    print(f"  Wavelength range: [{source_jax.minwave():.0f}, {source_jax.maxwave():.0f}] Å")

    # Calculate test bandflux and compare with sncosmo
    print("\nValidating against sncosmo...")
    test_phase = 0.0
    test_band = 'bessellb'

    # JAX version
    params_jax = {'amplitude': 1.0}
    flux_jax = source_jax.bandflux(params_jax, test_band, test_phase)

    # sncosmo version
    hsiao.set(amplitude=1.0)
    flux_snc = hsiao.bandflux(test_band, test_phase)

    print(f"  Phase: {test_phase:.1f} days, Band: {test_band}")
    print(f"  JAX flux:     {flux_jax:.6e}")
    print(f"  sncosmo flux: {flux_snc:.6e}")
    print(f"  Difference:   {abs(flux_jax - flux_snc)/flux_snc * 100:.4f}%")

    if abs(flux_jax - flux_snc)/flux_snc < 0.01:
        print("  ✓ Agreement < 1% !")

    return source_jax, phases, wavelengths, flux_grid


# ============================================================================
# Example 2: Load SED from Text Files
# ============================================================================

def example_2_load_from_files():
    """Demonstrate loading SED data from text files."""
    print("\n" + "=" * 70)
    print("Example 2: Creating SED from File Format")
    print("=" * 70)

    # Create example data structure (simulating what you'd load from files)
    print("\nSimulating SED data loaded from text files...")
    print("(In practice, you'd load from .txt/.dat files with observed spectra)")

    # Example: 5 epochs of spectra
    phases = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])  # Days relative to peak
    wavelengths = np.arange(3500.0, 8500.0, 5.0)  # Angstroms

    # Simulate realistic Type Ia-like SED evolution
    flux_grid = np.zeros((len(phases), len(wavelengths)))

    for i, phase in enumerate(phases):
        # Simple analytical model: blackbody that cools and dims
        temperature = 10000 / (1.0 + phase/20.0)  # K, cooling
        luminosity_factor = np.exp(-0.5 * (phase / 15.0)**2)  # Dimming

        # Wien's approximation for peak wavelength
        lambda_peak = 2.898e7 / temperature  # Angstroms

        # Gaussian-like SED shape
        sed_shape = np.exp(-0.5 * ((wavelengths - lambda_peak) / 800.0)**2)
        flux_grid[i, :] = sed_shape * luminosity_factor * 1e-15

    print(f"\nSimulated data:")
    print(f"  Epochs: {len(phases)} ({phases[0]:.1f} to {phases[-1]:.1f} days)")
    print(f"  Wavelength coverage: {wavelengths[0]:.0f} - {wavelengths[-1]:.0f} Å")
    print(f"  Spectral resolution: {wavelengths[1] - wavelengths[0]:.1f} Å")

    # This is how you'd load from actual files:
    print("\nExample file loading code:")
    print("  # Load phase values")
    print("  phases = np.loadtxt('spectra/phases.txt')")
    print("  ")
    print("  # Load wavelength grid")
    print("  wavelengths = np.loadtxt('spectra/wavelengths.txt')")
    print("  ")
    print("  # Load flux grid (one file per epoch)")
    print("  flux_grid = []")
    print("  for phase in phases:")
    print("      filename = f'spectra/spectrum_day{int(phase):+03d}.txt'")
    print("      flux = np.loadtxt(filename)")
    print("      flux_grid.append(flux)")
    print("  flux_grid = np.array(flux_grid)")

    # Create TimeSeriesSource
    print("\nCreating TimeSeriesSource...")
    source = TimeSeriesSource(
        phases, wavelengths, flux_grid,
        zero_before=True,
        time_spline_degree=3,
        name='custom_sed_from_files',
        version='1.0'
    )

    print(f"✓ Created TimeSeriesSource from 'file' data")

    # Calculate example light curve
    print("\nGenerating B-band light curve...")
    lc_phases = np.linspace(-15, 15, 50)
    lc_fluxes = source.bandflux({'amplitude': 1.0}, 'bessellb', lc_phases)

    print(f"  Phase range: {lc_phases[0]:.1f} to {lc_phases[-1]:.1f} days")
    print(f"  Flux range: {np.min(lc_fluxes):.3e} to {np.max(lc_fluxes):.3e}")

    return source, phases, wavelengths, flux_grid


# ============================================================================
# Example 3: Create SED from Analytical Model
# ============================================================================

def example_3_analytical_model():
    """Create SED from analytical/theoretical model."""
    print("\n" + "=" * 70)
    print("Example 3: Analytical SED Model")
    print("=" * 70)

    print("\nCreating simple expanding photosphere model...")

    # Define grids
    phases = np.linspace(-20, 40, 80)
    wavelengths = np.linspace(2500, 9000, 500)

    # Create mesh grids for vectorised calculation
    phase_grid, wave_grid = np.meshgrid(phases, wavelengths, indexing='ij')

    # Simple physical model parameters
    t_explosion = 0.0  # Explosion time (days)
    t0 = 17.0  # Characteristic timescale (days)
    L0 = 1.0e43  # Peak luminosity (erg/s)
    T0 = 10000  # Initial photospheric temperature (K)

    # Time-dependent luminosity (rise and decline)
    time_since_explosion = phase_grid - t_explosion
    luminosity = L0 * (time_since_explosion / t0)**2 * np.exp(-time_since_explosion / t0)
    luminosity = np.maximum(luminosity, 0)  # No negative luminosity

    # Time-dependent temperature (cooling)
    temperature = T0 * (1 + time_since_explosion / 30.0)**(-0.5)

    # Blackbody SED
    h = 6.626e-27  # Planck's constant (erg s)
    c = 3.0e10  # Speed of light (cm/s)
    k_B = 1.381e-16  # Boltzmann constant (erg/K)

    # Convert wavelength to cm
    wave_cm = wave_grid * 1e-8

    # Planck function (erg/s/cm²/Å/sr)
    B_lambda = (2 * h * c**2 / wave_cm**5) / (np.exp(h * c / (wave_cm * k_B * temperature)) - 1)

    # Convert to flux (assume distance and solid angle)
    distance = 10 * 3.086e18  # 10 pc in cm
    radius = 1e14  # Photosphere radius (cm) - roughly constant
    solid_angle = np.pi * radius**2 / distance**2

    # Final flux grid (erg/s/cm²/Å)
    flux_grid = B_lambda * solid_angle * (luminosity / L0) * 1e8  # Convert /cm to /Å

    print(f"Model parameters:")
    print(f"  Peak luminosity: {L0:.2e} erg/s")
    print(f"  Initial temperature: {T0:.0f} K")
    print(f"  Timescale: {t0:.1f} days")

    print(f"\nGrid dimensions:")
    print(f"  Phases: {len(phases)} points ({phases[0]:.1f} to {phases[-1]:.1f} days)")
    print(f"  Wavelengths: {len(wavelengths)} points ({wavelengths[0]:.0f} to {wavelengths[-1]:.0f} Å)")
    print(f"  Flux range: {np.min(flux_grid):.3e} to {np.max(flux_grid):.3e} erg/s/cm²/Å")

    # Create TimeSeriesSource
    source = TimeSeriesSource(
        phases, wavelengths, flux_grid,
        zero_before=True,
        time_spline_degree=3,
        name='analytical_expanding_photosphere',
        version='1.0'
    )

    print(f"\n✓ Created TimeSeriesSource from analytical model")

    return source, phases, wavelengths, flux_grid


# ============================================================================
# Example 4: Fit Custom SED to Photometric Data
# ============================================================================

def example_4_fit_to_photometry(source):
    """Fit custom SED to simulated photometric observations."""
    print("\n" + "=" * 70)
    print("Example 4: Fitting Custom SED to Photometry")
    print("=" * 70)

    print("\nGenerating synthetic photometric observations...")

    # Create "true" light curve with amplitude=2.0
    true_amplitude = 2.0
    obs_phases = jnp.array([-10, -5, 0, 2, 5, 10, 15])
    obs_bands = ['bessellb', 'bessellb', 'bessellv', 'bessellv', 'bessellr', 'bessellr', 'besselli']

    # Generate true fluxes
    true_fluxes = []
    for phase, band in zip(obs_phases, obs_bands):
        flux = source.bandflux({'amplitude': true_amplitude}, band, phase)
        true_fluxes.append(flux)
    true_fluxes = jnp.array(true_fluxes)

    # Add noise (5% uncertainties)
    np.random.seed(42)
    flux_errors = 0.05 * np.abs(true_fluxes)  # Use absolute value
    flux_errors = np.maximum(flux_errors, 1e-10)  # Ensure non-zero errors
    obs_fluxes = true_fluxes + np.random.normal(0, np.array(flux_errors))
    obs_fluxes = jnp.array(obs_fluxes)
    flux_errors = jnp.array(flux_errors)

    print(f"\nObservations:")
    print(f"  {'Phase':>6} {'Band':>10} {'Flux':>12} {'Error':>12}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*12}")
    # Convert to numpy for printing
    obs_fluxes_np = np.array(obs_fluxes)
    flux_errors_np = np.array(flux_errors)
    for i, (phase, band) in enumerate(zip(np.array(obs_phases), obs_bands)):
        flux_val = obs_fluxes_np[i].item() if hasattr(obs_fluxes_np[i], 'item') else obs_fluxes_np[i]
        err_val = flux_errors_np[i].item() if hasattr(flux_errors_np[i], 'item') else flux_errors_np[i]
        print(f"  {phase:6.1f} {band:>10} {flux_val:12.6e} {err_val:12.6e}")

    # Set up optimised mode for fitting
    print("\nSetting up optimised mode for fitting...")
    unique_bands = ['bessellb', 'bessellv', 'bessellr', 'besselli']
    bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)

    # Create band indices mapping
    band_to_idx = {band: i for i, band in enumerate(unique_bands)}
    band_indices = jnp.array([band_to_idx[band] for band in obs_bands])

    print(f"  Unique bands: {unique_bands}")
    print(f"  Band indices: {band_indices}")

    # Define negative log-likelihood (for minimisation)
    @jax.jit
    def neg_log_likelihood(amplitude):
        """Negative log-likelihood for fitting."""
        params = {'amplitude': amplitude}
        model_fluxes = source.bandflux(
            params, None, obs_phases,
            band_indices=band_indices,
            bridges=bridges,
            unique_bands=unique_bands
        )

        chi2 = jnp.sum(((obs_fluxes - model_fluxes) / flux_errors)**2)
        return 0.5 * chi2

    # Fit using simple grid search
    print("\nFitting amplitude parameter...")
    test_amplitudes = jnp.linspace(0.5, 4.0, 100)
    chi2_values = jnp.array([neg_log_likelihood(a) for a in test_amplitudes])

    best_idx = jnp.argmin(chi2_values)
    best_amplitude = test_amplitudes[best_idx]
    best_chi2 = chi2_values[best_idx]

    print(f"\nFit results:")
    print(f"  True amplitude: {true_amplitude:.4f}")
    print(f"  Best fit amplitude: {best_amplitude:.4f}")
    print(f"  Difference: {abs(best_amplitude - true_amplitude):.4f}")
    print(f"  χ²/dof: {best_chi2:.2f} / {len(obs_fluxes) - 1}")
    print(f"  Reduced χ²: {best_chi2 / (len(obs_fluxes) - 1):.2f}")

    return best_amplitude, test_amplitudes, chi2_values


# ============================================================================
# Example 5: Compare Custom SED with SALT3
# ============================================================================

def example_5_compare_with_salt3():
    """Compare custom SED with SALT3Source."""
    print("\n" + "=" * 70)
    print("Example 5: Comparing Custom SED with SALT3")
    print("=" * 70)

    if not HAS_SNCOSMO:
        print("Skipping - requires sncosmo for Hsiao template\n")
        return

    print("\nCreating Hsiao TimeSeriesSource...")
    hsiao = sncosmo.get_source('hsiao')
    phases = np.arange(-10.0, 40.0, 1.0)
    wavelengths = np.arange(3000.0, 9000.0, 10.0)

    flux_grid = np.zeros((len(phases), len(wavelengths)))
    for i, phase in enumerate(phases):
        flux_grid[i, :] = hsiao._flux(phase, wavelengths)

    source_hsiao = TimeSeriesSource(phases, wavelengths, flux_grid,
                                    zero_before=True, time_spline_degree=3)

    print("✓ Hsiao TimeSeriesSource created")

    # Create SALT3Source
    print("\nCreating SALT3Source...")
    source_salt3 = SALT3Source()
    print("✓ SALT3Source created")

    # Calculate light curves
    lc_phases = np.linspace(-10, 30, 50)
    bands = ['bessellb', 'bessellv', 'bessellr']

    print(f"\nGenerating light curves in {len(bands)} bands...")

    # Hsiao light curves
    params_hsiao = {'amplitude': 1.0}
    lc_hsiao = {}
    for band in bands:
        lc_hsiao[band] = source_hsiao.bandflux(params_hsiao, band, lc_phases)

    # SALT3 light curves (typical Type Ia)
    params_salt3 = {
        'z': 0.0,
        't0': 0.0,
        'x0': 1.0e-5,  # Scaled for similar flux level
        'x1': 0.0,     # Average stretch
        'c': 0.0       # No colour
    }
    lc_salt3 = {}
    for band in bands:
        lc_salt3[band] = source_salt3.bandflux(params_salt3, band, lc_phases)

    # Print comparison
    print(f"\n{'Band':>10} {'Peak Flux (Hsiao)':>20} {'Peak Flux (SALT3)':>20} {'Shape Similarity':>20}")
    print("-" * 75)
    for band in bands:
        peak_hsiao = np.max(lc_hsiao[band])
        peak_salt3 = np.max(lc_salt3[band])

        # Normalised correlation as similarity measure
        norm_hsiao = lc_hsiao[band] / peak_hsiao
        norm_salt3 = lc_salt3[band] / peak_salt3
        similarity = np.corrcoef(norm_hsiao, norm_salt3)[0, 1]

        print(f"{band:>10} {peak_hsiao:20.6e} {peak_salt3:20.6e} {similarity:20.4f}")

    print("\nNote: Hsiao is a simpler template, SALT3 includes color/stretch variations")

    return source_hsiao, source_salt3, lc_phases, lc_hsiao, lc_salt3


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Custom SED Example: Real Spectral Energy Distributions".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    # Example 1: Load Hsiao template
    result_1 = example_1_hsiao_template()

    # Example 2: Load from files (simulated)
    result_2 = example_2_load_from_files()

    # Example 3: Analytical model
    result_3 = example_3_analytical_model()
    if result_3 is not None:
        source_analytical = result_3[0]

        # Example 4: Fit to photometry
        example_4_fit_to_photometry(source_analytical)

    # Example 5: Compare with SALT3
    example_5_compare_with_salt3()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Try loading your own spectral data files")
    print("  2. Experiment with different interpolation degrees")
    print("  3. Fit to real photometric observations")
    print("  4. Use in MCMC/nested sampling workflows")
    print("")
