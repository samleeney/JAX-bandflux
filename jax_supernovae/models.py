import jax.numpy as jnp
import sncosmo
from .core import trapz, HC_ERG_AA, get_bandpass, get_magsystem

class Model:
    """A supernova model that can be evaluated at arbitrary times and wavelengths."""
    def __init__(self, source=None):
        self.source = source
        self.parameters = {'z': 0.0}  # Initialize with default redshift
        self.wave = None  # Will be set by source

    def _compute_bandflux(self, time, wave_obs, flux, transmission):
        """Internal function to compute bandflux that can be JIT compiled."""
        # Convert from erg/s/cm^2/Å to photons/s/cm^2/Å
        flux_photons = flux * wave_obs[None, :] / HC_ERG_AA
        
        # Integrate over wavelength
        integrand = flux_photons * transmission[None, :]
        return trapz(integrand, wave_obs, axis=1)

    def bandflux(self, band, time, zp=None, zpsys=None):
        """Flux through the given bandpass(es) at the given time(s)."""
        # Ensure inputs are JAX arrays
        time = jnp.atleast_1d(time)
        
        # Get bandpass
        bandpass = get_bandpass(band)
        wave = jnp.array(bandpass.wave)
        
        # Apply redshift
        redshift = self.parameters['z']
        wave_obs = wave * (1 + redshift)
        
        # Get transmission at observed wavelengths
        transmission = bandpass(wave_obs)
        
        # Evaluate source flux (in erg/s/cm^2/Å)
        flux = self.flux(time[:, None], wave_obs[None, :])
        
        # Compute integrated flux (in photons/s/cm^2)
        flux = self._compute_bandflux(time, wave_obs, flux, transmission)
        
        # Apply zeropoint scaling if requested
        if zp is not None:
            if zpsys is None:
                raise ValueError("zpsys must be provided if zp is not None")
            magsys = get_magsystem(zpsys)
            zp_flux = magsys.zpbandflux(bandpass)
            flux = flux / zp_flux * 10**(-0.4 * (zp - 25))
            
        return flux

    def flux(self, time, wave):
        """Calculate flux at the given times and wavelengths."""
        if self.source is None:
            raise ValueError("No source model set")
        return self.source.flux(time, wave) 