import jax.numpy as jnp
import numpy as np

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
    def __init__(self, wave, trans):
        """Initialize bandpass with wavelength and transmission arrays."""
        self.wave = jnp.asarray(wave, dtype=jnp.float64)
        self.trans = jnp.asarray(trans, dtype=jnp.float64)
        
    def __call__(self, wave_obs):
        """Interpolate transmission at given wavelengths."""
        return np.interp(wave_obs, self.wave, self.trans)
        
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
        """Divide the range between low and high into uniform bins with spacing less than or equal to target_spacing."""
        range_diff = high - low
        spacing = range_diff / int(np.ceil(range_diff / target_spacing))
        grid = np.arange(low + 0.5 * spacing, high, spacing)
        return grid, spacing

    def bandflux(self, band, time, zp=None, zpsys=None):
        """Compute synthetic photometry for a bandpass.
        
        Parameters
        ----------
        band : Bandpass
            The bandpass to compute synthetic photometry for.
        time : array_like
            Time(s) at which to evaluate the flux.
        zp : float or None
            If given, zeropoint to scale flux to (must also supply zpsys).
        zpsys : str or None
            Name of a magnitude system in the registry, specifying the system
            that zp is in.
            
        Returns
        -------
        array_like
            Flux in photons / s / cm^2.
        """
        # Get the bandpass data
        if not isinstance(band, Bandpass):
            raise ValueError("band must be a Bandpass object")
        
        # Convert time to numpy array
        time = np.array(time)
        
        # Get rest-frame time and wavelength
        z = self.parameters.get('z', 0.0)
        t_rest = time / (1.0 + z)
        
        # Use bandpass wavelength grid directly
        wave_obs = band.wave
        wave_rest = wave_obs / (1.0 + z)
        
        # Get the flux in the rest frame
        rest_flux = self.flux(t_rest[:, None], wave_rest[None, :])
        
        # Convert to numpy for integration
        wave_obs_np = np.array(wave_obs)
        trans_np = np.array(band.trans)
        rest_flux_np = np.array(rest_flux).squeeze()
        
        # Scale by a^2 to conserve bolometric luminosity
        a = 1.0 / (1.0 + z)
        
        # Integrate using trapz
        bandflux = np.array([np.sum(wave_obs_np * trans_np * f * a * a) * (wave_obs_np[1] - wave_obs_np[0]) / HC_ERG_AA for f in rest_flux_np])
        print("bandflux is ", bandflux)
        print("bandflux[0] is ", bandflux[0])
        jk
        
        return bandflux[0] if len(bandflux) == 1 else bandflux

def get_magsystem(name):
    """Get a MagSystem object by name."""
    if name != 'ab':
        raise ValueError("Only 'ab' magnitude system is supported")
    return MagSystem(name) 