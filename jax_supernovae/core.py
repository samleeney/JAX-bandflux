import jax.numpy as jnp

# Constants
HC_ERG_AA = 1.9865e-8  # erg * Angstrom
C_AA_PER_S = 2.99792458e18  # Angstrom/s

def trapz(y, x, axis=-1):
    """Trapezoidal integration along a given axis."""
    d = jnp.diff(x)
    y_avg = (y[..., 1:] + y[..., :-1]) / 2.0
    return jnp.sum(d * y_avg, axis=axis)

class Bandpass:
    def __init__(self, wave, trans, normalize=True):
        """Create a Bandpass object.
        
        Parameters
        ----------
        wave : array_like
            Wavelength array in Angstroms.
        trans : array_like
            Transmission array.
        normalize : bool, optional
            If True, normalize the transmission to have a maximum of 1.0.
        """
        self.wave = jnp.array(wave)
        self.trans = jnp.array(trans)
        
        if normalize:
            self.trans = self.trans / jnp.max(self.trans)
            
    def __call__(self, wave):
        """Evaluate the transmission at the given wavelengths."""
        return jnp.interp(wave, self.wave, self.trans, left=0.0, right=0.0)

class MagSystem:
    def __init__(self, name='ab'):
        """Create a MagSystem object.
        
        Parameters
        ----------
        name : str, optional
            Name of the magnitude system. Currently only 'ab' is supported.
        """
        self.name = name
        # AB system has constant F_nu = 3631 Jy = 3631 * 10^-23 erg/s/cm^2/Hz
        self.zp_flux_density = 3631e-23  # erg/s/cm^2/Hz
        
    def zpbandflux(self, band):
        """Calculate the flux of a zero magnitude object in photons/s/cm^2.

        Parameters
        ----------
        band : Bandpass
            The bandpass to calculate the zeropoint flux for.

        Returns
        -------
        float
            The zeropoint flux in photons/s/cm^2.
        """
        wave = band.wave
        trans = band(wave)

        # Convert F_nu (3631 Jy) to photon flux
        # F_nu = 3631e-23 erg/s/cm^2/Hz
        # Convert to photon flux using wave * trans / HC_ERG_AA
        photon_flux = self.zp_flux_density * wave * trans / HC_ERG_AA

        # Integrate over wavelength
        zp_flux = trapz(photon_flux, wave)

        return zp_flux

def get_bandpass(name):
    """Get a bandpass by name."""
    import sncosmo
    bp = sncosmo.get_bandpass(name)
    return Bandpass(bp.wave, bp.trans)

def get_magsystem(name):
    """Get a magnitude system by name."""
    if name.lower() != 'ab':
        raise ValueError("Only 'ab' magnitude system is currently supported")
    return MagSystem(name) 