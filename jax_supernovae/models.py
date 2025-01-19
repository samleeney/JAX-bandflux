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

        # Scale by a to convert from rest frame to observer frame
        # Note: We scale by a to conserve bolometric luminosity
        # The wavelength scaling is handled by the integration over the bandpass
        return rest_flux * a

    def bandflux(self, band, time, zp=None, zpsys=None):
        """Compute synthetic photometry for a bandpass.

        Parameters
        ----------
        band : str or Bandpass
            Bandpass object or name of registered bandpass.
        time : array_like
            Times at which to evaluate the flux.
        zp : float or None, optional
            If given, zeropoint to scale the bandpass to.
        zpsys : str or None, optional
            Name of a magnitude system. Required if zp is given.

        Returns
        -------
        array_like
            Flux in photons / s / cm^2.
        """
        # Get bandpass object if a string is provided
        if isinstance(band, str):
            band = get_bandpass(band)
        elif not isinstance(band, Bandpass):
            raise ValueError("band must be a Bandpass object or a string")

        # Convert time to numpy array and ensure it's at least 1D
        time = np.atleast_1d(time)

        # Get rest-frame time and wavelength
        z = self.parameters['z']
        a = 1.0 / (1.0 + z)

        # Use bandpass wavelength grid directly
        wave_obs = band.wave
        dwave = wave_obs[1] - wave_obs[0]  # Assuming uniform spacing

        # Get bandpass transmission
        trans = band.trans

        # Convert to numpy arrays for integration
        wave_obs_np = np.array(wave_obs)
        trans_np = np.array(trans)

        # First multiply transmission by wavelength (following SNCosmo's approach)
        tmp = trans_np * wave_obs_np

        # Get rest-frame flux for each time and wavelength
        rest_flux = np.zeros((len(time), len(wave_obs)))
        for i, t in enumerate(time):
            rest_flux[i] = self._flux_with_redshift(t, wave_obs)

        # Calculate bandflux using SNCosmo's formula
        bandflux = np.sum(rest_flux * tmp[None, :], axis=1) * dwave / HC_ERG_AA

        # Apply zeropoint scaling if provided
        if zp is not None:
            if zpsys is None:
                raise ValueError('zpsys must be given if zp is not None')
            ms = get_magsystem(zpsys)
            zp_bandflux = ms.zpbandflux(band)
            zpnorm = 10. ** (0.4 * zp) / zp_bandflux
            bandflux *= zpnorm

        # Return scalar if input was scalar
        return float(bandflux) if len(time) == 1 else bandflux