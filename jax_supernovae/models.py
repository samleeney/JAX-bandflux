import jax.numpy as jnp
import numpy as np
from jax_supernovae.core import HC_ERG_AA, Bandpass, get_magsystem
from jax_supernovae.bandpasses import get_bandpass

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
    grid = np.arange(low + 0.5 * spacing, high, spacing)
    return grid, spacing

class Model:
    def __init__(self, source=None):
        self.source = source
        self.parameters = {}
        self._flux = None

    def _flux_with_redshift(self, time, wave):
        """Calculate flux with redshift scaling.

        Parameters
        ----------
        time : array_like
            Observer-frame time(s) in days.
        wave : array_like
            Observer-frame wavelength(s) in Angstroms.

        Returns
        -------
        array_like
            Observer-frame flux values in ergs/s/cm^2/Angstrom.
        """
        # Convert to rest frame
        z = self.parameters['z']
        t0 = self.parameters['t0']
        a = 1. / (1. + z)  # scale factor
        restphase = (time - t0) * a  # rest-frame phase
        restwave = wave * a  # rest-frame wavelength

        # Get rest-frame flux
        rest_flux = self._flux(restphase, restwave)

        # Scale by a^2 to convert from rest frame to observer frame
        return rest_flux * a * a

    def bandflux(self, band, time, zp=None, zpsys=None):
        """Compute synthetic photometry in a given bandpass.

        Parameters
        ----------
        band : str or Bandpass
            Bandpass object or name of registered bandpass.
        time : array_like
            Observer-frame times.
        zp : array_like, optional
            If given, zeropoint to scale flux to (must include units).
        zpsys : str or MagSystem
            If given, magnitude system to scale flux to.

        Returns
        -------
        flux : array_like
            Flux in photons / s / cm^2.
        """
        # Get bandpass object if a string is provided
        if isinstance(band, str):
            band = get_bandpass(band)

        # Convert time to numpy array
        time = np.asarray(time)

        # Get wavelength range and integration grid
        wave_min = band.minwave()
        wave_max = band.maxwave()
        wave_obs, dwave = integration_grid(wave_min, wave_max, 5.0)  # Use SNCosmo's default spacing

        # Get transmission at each wavelength
        trans = band(wave_obs)

        # Calculate rest frame flux
        rest_flux = self._flux_with_redshift(time[:, None], wave_obs[None, :])

        # Convert to numpy arrays for integration
        wave_obs_np = np.array(wave_obs)
        trans_np = np.array(trans)
        rest_flux_np = np.array(rest_flux)

        # Calculate weights for integration (wave * trans / HC_ERG_AA)
        weights = wave_obs_np * trans_np / HC_ERG_AA

        # Integrate over wavelength for each time
        bandflux = np.array([np.sum(weights * f) * dwave for f in rest_flux_np])

        # Scale by zeropoint if provided
        if zp is not None:
            if zpsys is None:
                raise ValueError('zpsys must be given if zp is not None')
            ms = get_magsystem(zpsys)
            zp_bandflux = ms.zpbandflux(band)
            zpnorm = 10.**(0.4 * zp) / zp_bandflux
            bandflux *= zpnorm

        return bandflux