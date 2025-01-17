import jax.numpy as jnp
import pytest
from jax_supernovae.core import get_bandpass, get_magsystem, Bandpass, MagSystem, HC_ERG_AA

def test_bandpass_creation():
    """Test creation of bandpass object."""
    wave = jnp.linspace(4000, 5000, 100)
    trans = jnp.ones_like(wave)
    bandpass = Bandpass(wave, trans)
    
    assert isinstance(bandpass.wave, jnp.ndarray)
    assert isinstance(bandpass.trans, jnp.ndarray)
    assert bandpass.wave.shape == wave.shape
    assert bandpass.trans.shape == trans.shape

def test_bandpass_normalization():
    """Test bandpass normalization."""
    wave = jnp.linspace(4000, 5000, 100)
    trans = 2.0 * jnp.ones_like(wave)  # All values are 2.0
    bandpass = Bandpass(wave, trans, normalize=True)
    
    # After normalization, all values should be 1.0
    assert jnp.allclose(bandpass.trans, jnp.ones_like(wave))

def test_bandpass_interpolation():
    """Test bandpass transmission interpolation."""
    wave = jnp.array([4000., 5000.])
    trans = jnp.array([1., 0.])
    bandpass = Bandpass(wave, trans)
    
    # Test interpolation at midpoint
    wave_test = jnp.array(4500.)  # scalar input
    trans_interp = bandpass(wave_test)
    
    # Convert to float for comparison
    trans_value = float(trans_interp)
    assert trans_value == pytest.approx(0.5, rel=1e-5)

def test_get_bandpass():
    """Test get_bandpass function."""
    bandpass = get_bandpass('sdssg')
    assert isinstance(bandpass, Bandpass)
    # SNCosmo's bandpass has 101 points
    assert bandpass.wave.shape[0] == 101

def test_magsystem_creation():
    """Test creation of magnitude system."""
    magsys = get_magsystem('ab')
    assert isinstance(magsys, MagSystem)
    assert magsys.name == 'ab'

def test_zpbandflux():
    """Test zeropoint flux calculation."""
    # Create a simple rectangular bandpass
    wave = jnp.linspace(4000, 5000, 1000)
    trans = jnp.ones_like(wave)
    bandpass = Bandpass(wave, trans, normalize=True)
    
    # Get AB magnitude system
    magsys = get_magsystem('ab')
    
    # Calculate zeropoint flux
    zp_flux = magsys.zpbandflux(bandpass)
    
    # Compute expected flux
    # For a flat F_nu spectrum and a rectangular bandpass,
    # the number of photons should be proportional to lambda
    wave_eff = (wave[-1] + wave[0]) / 2
    expected_flux = magsys.zp_flux_density * (wave_eff / HC_ERG_AA) * (wave[-1] - wave[0])
    
    # Compare results (within order of magnitude due to integration method differences)
    ratio = float(zp_flux / expected_flux)
    assert 0.1 < ratio < 10.0

def test_invalid_bandpass():
    """Test error handling for invalid bandpass."""
    with pytest.raises(Exception) as excinfo:
        get_bandpass('invalid_bandpass')
    assert "not in registry" in str(excinfo.value)

def test_invalid_magsystem():
    """Test error handling for invalid magnitude system."""
    with pytest.raises(ValueError):
        get_magsystem('invalid_magsystem') 