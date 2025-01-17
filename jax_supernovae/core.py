import jax.numpy as jnp
import sncosmo

# Constants
H_ERG_S = 6.626068e-27   # Planck's constant in erg seconds
C_AA_PER_S = 2.99792458e18  # Speed of light in angstroms per second
HC_ERG_AA = H_ERG_S * C_AA_PER_S  # erg * angstrom

def trapz(y, x, axis=-1):
    """JAX implementation of numpy's trapz integration."""
    d = jnp.diff(x)
    y_avg = (y[..., 1:] + y[..., :-1]) / 2.0
    return jnp.sum(d * y_avg, axis=axis)

class Bandpass:
    def __init__(self, wave, trans, normalize=True):
        """Create a Bandpass object.
        
        Parameters
        ----------
        wave : array_like
            Wavelength values in angstroms.
        trans : array_like
            Transmission values.
        normalize : bool, optional
            If True, normalize transmission to have maximum value of 1.0.
        """
        self.wave = jnp.asarray(wave)
        self.trans = jnp.asarray(trans)
        
        if normalize:
            self.trans = self.trans / jnp.max(self.trans)
            
    def __call__(self, wave_obs):
        """Return transmission at given wavelengths using linear interpolation."""
        return jnp.interp(wave_obs, self.wave, self.trans, left=0.0, right=0.0)
        
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
        
def get_bandpass(name):
    """Get a Bandpass object by name from the registry."""
    try:
        band = sncosmo.get_bandpass(name)
        return Bandpass(band.wave, band.trans)
    except Exception as e:
        raise Exception(f"Bandpass {name} not in registry") from e
        
def get_magsystem(name):
    """Get a MagSystem object by name."""
    if name != 'ab':
        raise ValueError("Only 'ab' magnitude system is supported")
    return MagSystem(name) 