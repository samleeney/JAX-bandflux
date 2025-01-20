import jax
import jax.numpy as jnp
import numpy as np
from jax_supernovae.core import HC_ERG_AA, Bandpass, get_magsystem, MODEL_BANDFLUX_SPACING
from archive.bandpasses import get_bandpass

def integration_grid(low, high, target_spacing):
    """Divide the range between low and high into uniform bins with spacing
    less than or equal to target_spacing and return the bin midpoints and
    the actual spacing.

    Parameters
    ----------
    low : float
        Lower bound of range.
    high : float
        Upper bound of range.
    target_spacing : float
        Target spacing between bins.

    Returns
    -------
    grid : array_like
        Bin midpoints.
    spacing : float
        Actual spacing used.
    """
    range_diff = high - low
    spacing = range_diff / int(np.ceil(range_diff / target_spacing))
    grid = jnp.arange(low + 0.5 * spacing, high, spacing)
    return grid, spacing

class Model:
    """A supernova model.

    Parameters
    ----------
    source : str or None, optional
        Name of source model to use. Default is None.
    """
    def __init__(self, source=None):
        self.source = source
        self.parameters = {}
        self.wave = None
        self.flux = None

    def bandflux(self, b, times, zp=None, zpsys=None):
        """Calculate flux through a bandpass at given times.
        
        Parameters
        ----------
        b : str or Bandpass
            Name of bandpass or Bandpass object
        times : array_like
            Times at which to calculate flux
        zp : float or None, optional
            Zeropoint to scale flux to
        zpsys : str or None, optional
            Name of magnitude system that zp is in
        
        Returns
        -------
        array_like
            Flux in photons / s / cm^2
        """
        # Get the bandpass object
        b = get_bandpass(b)
        
        # Convert times to numpy array
        times = np.array(times)
        
        # Get rest-frame times and wavelengths
        z = self.parameters['z']
        t0 = self.parameters['t0']
        a = 1.0 / (1.0 + z)
        restphase = (times - t0) * a
        
        # Get bandpass wavelength grid
        wave = b.wave
        dwave = wave[1] - wave[0]  # Assuming uniform spacing
        restwave = wave * a
        
        # Get bandpass transmission
        trans = b.trans
        
        # Get rest-frame flux
        rest_flux = self.flux(restphase[:, None], restwave[None, :])
        
        # Calculate bandflux using SNCosmo's formula
        bandflux = np.sum(restwave * trans * rest_flux, axis=1) * dwave / HC_ERG_AA
        
        # Apply zeropoint scaling if provided
        if zp is not None:
            if zpsys is None:
                raise ValueError('zpsys must be given if zp is not None')
            ms = get_magsystem(zpsys)
            zp_bandflux = ms.zpbandflux(b)
            zpnorm = 10. ** (0.4 * zp) / zp_bandflux
            bandflux *= zpnorm
        
        return bandflux