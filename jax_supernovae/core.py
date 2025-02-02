"""Core functionality for JAX supernova models."""
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import splrep, splev
import jax
from functools import partial
from jax import vmap

# Constants - match SNCosmo exactly
HC_ERG_AA = 1.9865e-8  # h*c in erg*angstrom
C_AA_PER_S = 2.99792458e18  # speed of light in angstrom/sec

# Model constants
MODEL_BANDFLUX_SPACING = 5.0  # Wavelength spacing for bandflux integration

@partial(vmap, in_axes=(0, None, None))
def interp(x, xp, fp):
    """Linear interpolation for JAX arrays."""
    x = jnp.asarray(x)  # Don't reshape, preserve input shape
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    
    # Find indices of points to interpolate between
    i = jnp.searchsorted(xp, x)
    i = jnp.clip(i, 1, len(xp) - 1)
    
    # Get x and y values to interpolate between
    x0 = xp[i - 1]
    x1 = xp[i]
    y0 = fp[i - 1]
    y1 = fp[i]
    
    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)

# Use jax.numpy.trapezoid directly
trapz = jnp.trapezoid

class Bandpass:
    """Bandpass filter class."""
    
    def __init__(self, wave, trans):
        """Initialize bandpass with wavelength and transmission arrays."""
        self._wave = jnp.asarray(wave)
        self._trans = jnp.asarray(trans)
        self._minwave = float(jnp.min(wave))
        self._maxwave = float(jnp.max(wave))
    
    def __call__(self, wave):
        """Get interpolated transmission at given wavelengths."""
        wave = jnp.asarray(wave)  # Don't reshape, preserve input shape
        return interp(wave, self._wave, self._trans)
    
    def minwave(self):
        """Get minimum wavelength."""
        return self._minwave
    
    def maxwave(self):
        """Get maximum wavelength."""
        return self._maxwave
    
    @property
    def wave(self):
        """Get wavelength array."""
        return self._wave
    
    @property
    def trans(self):
        """Get transmission array."""
        return self._trans
