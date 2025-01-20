import pytest
import numpy as np
import jax.numpy as jnp
import sncosmo
from astropy import units as u
from jax_supernovae.core import Bandpass, HC_ERG_AA, MODEL_BANDFLUX_SPACING
from jax_supernovae.salt3nir import integration_grid

def test_integration_grid():
    """Test that integration_grid matches SNCosmo's grid."""
    # Create a test bandpass
    wave = np.array([3000., 4000., 5000., 6000., 7000.])
    trans = np.array([0., 0.5, 1.0, 0.5, 0.])
    band = Bandpass(wave, trans)
    
    # Create expected grid with spacing of 5.0
    spacing = 5.0
    nbin = int(np.ceil((wave[-1] - wave[0]) / spacing))
    expected_wave = wave[0] + (np.arange(nbin) + 0.5) * spacing
    
    # Get grid from bandpass
    grid_wave, grid_spacing = band.get_integration_grid()
    
    assert jnp.allclose(grid_wave, expected_wave), f"Grid wavelengths don't match"
    assert jnp.isclose(grid_spacing, spacing), f"Grid spacing doesn't match"

def test_transmission_interpolation():
    """Test that transmission interpolation matches SNCosmo."""
    # Create a test bandpass
    wave = np.array([3000., 4000., 5000., 6000., 7000.])
    trans = np.array([0., 0.5, 1.0, 0.5, 0.])
    band = Bandpass(wave, trans)
    snc_band = sncosmo.Bandpass(wave, trans)
    
    # Test points for interpolation
    test_wave = np.array([3500., 4500., 5500., 6500.])
    
    # Get interpolated transmission
    jax_trans = band(test_wave)
    snc_trans = snc_band(test_wave)
    
    assert jnp.allclose(jax_trans, snc_trans), f"Interpolated transmission doesn't match"

def test_transmission_normalization():
    """Test that transmission normalization matches SNCosmo."""
    # Create test bandpass with energy units
    wave = np.array([3000., 4000., 5000., 6000., 7000.])
    trans = np.array([0., 0.5, 1.0, 0.5, 0.])
    
    # Create bandpasses with different unit settings
    # For energy units, transmission is in units of (photons/erg)
    # This means we need to scale by wavelength first
    trans_energy = trans * (HC_ERG_AA / wave)
    trans_energy = trans_energy / np.max(trans_energy)  # Normalize after scaling
    
    band_energy = Bandpass(wave, trans, trans_unit='energy', normalize=True)
    band_photon = Bandpass(wave, trans, trans_unit='photon', normalize=True)
    
    # Create SNCosmo bandpass with dimensionless transmission
    snc_band = sncosmo.Bandpass(wave, trans)
    
    # Test points for interpolation
    test_wave = np.array([3500., 4500., 5500., 6500.])
    
    # Get interpolated transmission
    jax_trans_energy = band_energy(test_wave)
    jax_trans_photon = band_photon(test_wave)
    snc_trans = snc_band(test_wave)
    
    # Print values for debugging
    print(f"Energy transmission: {jax_trans_energy}")
    print(f"Photon transmission: {jax_trans_photon}")
    print(f"SNCosmo transmission: {snc_trans}")
    
    # For energy units, interpolate the pre-normalized energy transmission
    expected_trans = np.interp(test_wave, wave, trans_energy)
    
    assert jnp.allclose(jax_trans_energy, expected_trans), f"Energy transmission doesn't match"
    assert jnp.allclose(jax_trans_photon, snc_trans), f"Photon transmission doesn't match"

def test_transmission_trimming():
    """Test that transmission trimming matches SNCosmo."""
    # Create test bandpass with low transmission values
    wave = np.array([2000., 3000., 4000., 5000., 6000., 7000., 8000.])
    trans = np.array([0.0, 0.0001, 0.5, 1.0, 0.5, 0.0001, 0.0])  # More extreme test case
    
    # Create bandpasses with different trim levels
    band_trim = Bandpass(wave, trans, trim_level=0.001)
    band_notrim = Bandpass(wave, trans, trim_level=None)
    
    # Create SNCosmo bandpass with trimming
    snc_band = sncosmo.Bandpass(wave, trans, trim_level=0.001)
    
    # Get wavelength ranges as arrays
    wave_range_trim = jnp.array([band_trim.minwave(), band_trim.maxwave()])
    wave_range_notrim = jnp.array([band_notrim.minwave(), band_notrim.maxwave()])
    wave_range_snc = jnp.array([snc_band.minwave(), snc_band.maxwave()])
    
    print(f"Trimmed range: {wave_range_trim}")
    print(f"Untrimmed range: {wave_range_notrim}")
    print(f"SNCosmo range: {wave_range_snc}")
    
    # Check that trimming matches SNCosmo
    assert jnp.allclose(wave_range_trim, wave_range_snc), f"Trimmed wavelength range mismatch"
    # Check that untrimmed range is wider
    assert wave_range_notrim[0] < wave_range_trim[0], f"Untrimmed range should start earlier"
    assert wave_range_notrim[1] > wave_range_trim[1], f"Untrimmed range should end later"

def test_transmission_comparison():
    """Test that transmission handling matches exactly between SNCosmo and JAX."""
    # Create test bandpass with realistic values
    wave = np.array([4000., 5000., 6000., 7000., 8000.])
    trans = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
    
    # Create both bandpass objects
    jax_band = Bandpass(wave, trans)
    snc_band = sncosmo.Bandpass(wave, trans)
    
    # Create integration grid
    wave_grid, spacing = integration_grid(wave[0], wave[-1], MODEL_BANDFLUX_SPACING)
    
    # Get transmission on grid
    jax_trans = jax_band(wave_grid)
    snc_trans = snc_band(wave_grid)
    
    # Print detailed comparison
    print("\nTransmission Comparison:")
    print(f"{'Wave':>10s} {'JAX':>10s} {'SNCosmo':>10s} {'Ratio':>10s}")
    print("-" * 42)
    for i in range(0, len(wave_grid), len(wave_grid)//5):
        ratio = float(jax_trans[i]/snc_trans[i])
        print(f"{wave_grid[i]:10.1f} {jax_trans[i]:10.3f} {snc_trans[i]:10.3f} {ratio:10.3f}")
    
    # Test full arrays
    np.testing.assert_allclose(
        jax_trans, snc_trans, rtol=1e-10,
        err_msg="Transmission values don't match exactly"
    )
    
    # Test integration weights
    jax_weights = wave_grid * jax_trans
    snc_weights = wave_grid * snc_trans
    
    print("\nIntegration Weight Comparison:")
    print(f"{'Wave':>10s} {'JAX':>15s} {'SNCosmo':>15s} {'Ratio':>10s}")
    print("-" * 52)
    for i in range(0, len(wave_grid), len(wave_grid)//5):
        ratio = float(jax_weights[i]/snc_weights[i])
        print(f"{wave_grid[i]:10.1f} {jax_weights[i]:15.3e} {snc_weights[i]:15.3e} {ratio:10.3f}")
    
    np.testing.assert_allclose(
        jax_weights, snc_weights, rtol=1e-10,
        err_msg="Integration weights don't match exactly"
    ) 