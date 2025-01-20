"""Core functionality for JAX supernova models."""
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import splrep, splev

# Constants - match SNCosmo exactly
H_ERG_S = 6.626068e-27   # Planck's constant in erg seconds
C_AA_PER_S = 2.99792458e18  # Speed of light in angstroms per second
HC_ERG_AA = H_ERG_S * C_AA_PER_S  # erg * angstrom

# Model constants
MODEL_BANDFLUX_SPACING = 5.0  # Wavelength spacing for bandflux integration

def trapz(y, x, axis=-1):
    """JAX implementation of numpy's trapz integration."""
    d = jnp.diff(x)
    y_avg = (y[..., 1:] + y[..., :-1]) / 2.0
    return jnp.sum(d * y_avg, axis=axis)

class Bandpass:
    """A bandpass filter."""
    def __init__(self, wave, trans):
        self.wave = jnp.array(wave)
        self.trans = jnp.array(trans)
        self._setup_bandpass()
    
    def _setup_bandpass(self):
        """Set up bandpass properties."""
        # Normalize transmission
        self.trans = self.trans / jnp.max(self.trans)
        
        # Calculate effective wavelength
        weights = self.trans * self.wave
        self.wave_eff = jnp.sum(weights) / jnp.sum(self.trans)
        
        # Calculate wavelength range
        self.wave_min = jnp.min(self.wave)
        self.wave_max = jnp.max(self.wave)

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
        return splev(wave_obs, splrep(self.wave, self.trans, k=1), ext=1)
        
    def minwave(self):
        """Return minimum wavelength."""
        return float(self.wave_min)
        
    def maxwave(self):
        """Return maximum wavelength."""
        return float(self.wave_max)

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

def get_magsystem(name='ab'):
    """Get a magnitude system by name."""
    if name.lower() == 'ab':
        return ABMagSystem()
    else:
        raise ValueError(f"Unknown magnitude system: {name}")

class ABMagSystem:
    """The AB magnitude system."""
    def __init__(self):
        self.name = 'ab'
        self.zero_point_flux = 10**(48.6/-2.5)  # erg/s/cm^2/Hz
    
    def band_zero_point(self, bandpass):
        """Calculate the zero point for a given bandpass."""
        # Convert to per unit frequency
        flux_hz = self.zero_point_flux * bandpass.wave_eff**2 / C_AA_PER_S
        # Convert back to per unit wavelength
        return flux_hz * C_AA_PER_S / bandpass.wave_eff**2 