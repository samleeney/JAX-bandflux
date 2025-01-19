"""Core functionality for JAX supernova models."""
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import splrep, splev

# Constants - match SNCosmo exactly
H_ERG_S = 6.626068e-27   # Planck's constant in erg seconds
C_AA_PER_S = 2.99792458e18  # Speed of light in angstroms per second
HC_ERG_AA = H_ERG_S * C_AA_PER_S  # erg * angstrom

def trapz(y, x, axis=-1):
    """JAX implementation of numpy's trapz integration."""
    d = jnp.diff(x)
    y_avg = (y[..., 1:] + y[..., :-1]) / 2.0
    return jnp.sum(d * y_avg, axis=axis)

class Bandpass:
    def __init__(self, wave, trans, normalize=False):
        """Initialize a bandpass with arrays of wavelength and transmission.

        Parameters
        ----------
        wave : array_like
            Wavelength values in Angstroms.
        trans : array_like
            Transmission fraction at each wavelength.
        normalize : bool, optional
            If True, normalize transmission to 1.0 at peak (default: False).
        """
        wave = np.asarray(wave, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64)
        
        if wave.shape != trans.shape:
            raise ValueError('shape of wave and trans must match')
        if wave.ndim != 1:
            raise ValueError('only 1-d arrays supported')
            
        # Check that values are monotonically increasing
        if not np.all(np.diff(wave) > 0.):
            raise ValueError('wavelength values must be monotonically increasing')
            
        if normalize:
            trans = trans / np.max(trans)
            
        # Set up interpolation
        self._tck = splrep(wave, trans, k=1)
        
        self.wave = wave
        self.trans = trans
        
    def __call__(self, wave_obs):
        """Return interpolated transmission at given wavelengths.
        
        Parameters
        ----------
        wave_obs : array_like
            Wavelengths to evaluate transmission at.
            
        Returns
        -------
        trans : array_like
            Transmission fraction at each wavelength.
        """
        return splev(wave_obs, self._tck, ext=1)
        
    def minwave(self):
        """Return minimum wavelength."""
        return float(self.wave[0])
        
    def maxwave(self):
        """Return maximum wavelength."""
        return float(self.wave[-1])

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
        """Divide the range between `low` and `high` into uniform bins.

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
        grid : array
            Bin midpoints.
        spacing : float
            Actual spacing between bins.
        """
        range_diff = high - low
        spacing = range_diff / int(np.ceil(range_diff / target_spacing))
        grid = np.arange(low + 0.5 * spacing, high, spacing)

        return grid, spacing

    def bandflux(self, band, time, zp=None, zpsys=None):
        """Compute synthetic photometry for a bandpass.

        Parameters
        ----------
        band : Bandpass object
            Bandpass to compute synthetic photometry for.
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
        if not isinstance(band, Bandpass):
            raise ValueError("band must be a Bandpass object")

        # Convert time to numpy array
        time = np.array(time)

        # Get rest-frame time and wavelength
        z = self.parameters['z']
        a = 1.0 / (1.0 + z)
        t_rest = time * a

        # Use bandpass wavelength grid directly
        wave_obs = band.wave
        dwave = wave_obs[1] - wave_obs[0]  # Assuming uniform spacing
        wave_rest = wave_obs * a

        # Get bandpass transmission
        trans = band.trans

        # Convert to numpy arrays for integration
        wave_obs_np = np.array(wave_obs)
        trans_np = np.array(trans)

        # Get rest-frame flux
        rest_flux = self.flux(t_rest[:, None], wave_rest[None, :])
        rest_flux_np = np.array(rest_flux)

        # Calculate bandflux using SNCosmo's formula
        bandflux = np.sum(wave_rest * trans_np * rest_flux_np, axis=1) * dwave / HC_ERG_AA

        # Apply zeropoint scaling if provided
        if zp is not None:
            if zpsys is None:
                raise ValueError('zpsys must be given if zp is not None')
            ms = get_magsystem(zpsys)
            zp_bandflux = ms.zpbandflux(band)
            zpnorm = 10. ** (0.4 * zp) / zp_bandflux
            bandflux *= zpnorm

        return bandflux

def get_magsystem(name):
    """Get a MagSystem object by name."""
    if name != 'ab':
        raise ValueError("Only 'ab' magnitude system is supported")
    return MagSystem(name) 