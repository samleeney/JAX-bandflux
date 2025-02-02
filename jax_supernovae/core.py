"""Core functionality for JAX supernova models."""
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import splrep, splev
import jax
from functools import partial
import math
from .utils import interp
from .constants import HC_ERG_AA, C_AA_PER_S, MODEL_BANDFLUX_SPACING

# Use jax.numpy.trapezoid directly
trapz = jnp.trapezoid

class Bandpass:
    """Bandpass filter class."""
    
    def __init__(self, wave, trans, integration_spacing=MODEL_BANDFLUX_SPACING):
        """Initialize bandpass with wavelength and transmission arrays."""
        self._wave = jnp.asarray(wave)
        self._trans = jnp.asarray(trans)
        self._minwave = float(jnp.min(wave))
        self._maxwave = float(jnp.max(wave))
        
        # Pre-compute integration grid to match sncosmo exactly
        range_diff = self._maxwave - self._minwave
        n_steps = math.ceil(range_diff / integration_spacing)
        self._integration_spacing = range_diff / n_steps
        
        # Create grid starting at minwave + 0.5 * spacing
        self._integration_wave = jnp.linspace(
            self._minwave + 0.5 * self._integration_spacing,
            self._maxwave - 0.5 * self._integration_spacing,
            n_steps
        )
    
    def __call__(self, wave):
        """Get interpolated transmission at given wavelengths."""
        wave = jnp.asarray(wave)
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
        
    @property
    def integration_wave(self):
        """Get pre-computed integration wavelength grid."""
        return self._integration_wave
        
    @property
    def integration_spacing(self):
        """Get integration grid spacing."""
        return self._integration_spacing
