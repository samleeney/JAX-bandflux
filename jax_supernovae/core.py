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

    def get_integration_grid(self):
        """Get the wavelength grid used for integration.

        Returns
        -------
        wave : array_like
            Wavelength values in Angstroms.
        spacing : float
            Grid spacing in Angstroms.
        """
        spacing = MODEL_BANDFLUX_SPACING
        nbin = int(np.ceil((self.maxwave() - self.minwave()) / spacing))
        wave = jnp.arange(nbin) * spacing + self.minwave() + 0.5 * spacing
        return wave, spacing

class MagSystem:
    def __init__(self, name='ab'):
        """Create a MagSystem object.
        
        Parameters
        ----------
        name : str, optional
            Name of the magnitude system. Currently only 'ab' is supported.
        """
        if name != 'ab':
            raise ValueError("Only 'ab' magnitude system is supported")
        self.name = name
        self.zp_flux_density = 3631e-23  # erg/s/cm^2/Hz
        
    def zpbandflux(self, band):
        """Calculate the flux of a zero magnitude object in photons/s/cm^2.
        
        Parameters
        ----------
        band : Bandpass object
            Bandpass to calculate zeropoint flux for.
            
        Returns
        -------
        flux : float
            Flux in photons/s/cm^2.
        """
        wave = band.wave
        trans = band.trans
        
        # Convert F_nu (erg/s/cm^2/Hz) to photon flux
        flux_wave = self.zp_flux_density * (C_AA_PER_S / wave**2)
        photon_flux = flux_wave * wave * trans / HC_ERG_AA
        
        return trapz(photon_flux, wave)

    def integration_grid(self, low, high, target_spacing):
        """Divide the range between low and high into uniform bins.
        
        Parameters
        ----------
        low : float
            Lower bound of the range.
        high : float
            Upper bound of the range.
        target_spacing : float
            Target spacing between points.
            
        Returns
        -------
        wave : array_like
            Array of bin midpoints.
        spacing : float
            Actual spacing used.
        """

def get_magsystem(name):
    """Get a magnitude system by name.
    
    Parameters
    ----------
    name : str
        Name of the magnitude system. Currently only 'ab' is supported.
        
    Returns
    -------
    magsys : MagSystem
        The requested magnitude system.
    """
    return MagSystem(name)