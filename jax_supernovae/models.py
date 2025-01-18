import jax.numpy as jnp
import numpy as np
from jax_supernovae.core import HC_ERG_AA, Bandpass, get_magsystem

def integration_grid(wave_min, wave_max, spacing):
    """Create a wavelength grid for integration."""
    npt = int(np.ceil((wave_max - wave_min) / spacing)) + 1
    wave = wave_min + np.arange(npt) * spacing
    dwave = np.gradient(wave)
    return jnp.array(wave), jnp.array(dwave)

class Model:
    def __init__(self, source=None):
        self.source = source
        self.parameters = {}
        self.wave = None
        self.flux = None
        
    def bandflux(self, band, time, zp=None, zpsys=None):
        """Compute synthetic photometry for a bandpass.
        
        Args:
            band: Bandpass or name of bandpass.
            time: Time(s) in days.
            zp: Zeropoint to scale flux to (optional).
            zpsys: Magnitude system for zeropoint (optional).
            
        Returns:
            Flux in photons/s/cm^2.
        """
        # Get redshift parameter with default of 0.0
        z = self.parameters.get('z', 0.0)
        
        # Scale factor to conserve bolometric luminosity
        a = 1. / (1. + z)
        
        # Convert to rest frame time
        t_rest = (time - self.parameters.get('t0', 0.0)) * a
        
        # Get bandpass object and shift it to rest frame
        if not isinstance(band, Bandpass):
            raise ValueError("Only Bandpass objects are supported")
        b_rest = Bandpass(band.wave * a, band.trans)
        
        # Get wavelength grid for integration in rest frame
        wave_min = b_rest.minwave()
        wave_max = b_rest.maxwave()
        wave_rest, dwave = integration_grid(wave_min, wave_max, 0.1)  # Use smaller spacing
        
        # Evaluate flux at rest-frame wavelengths
        rest_flux = self.flux(t_rest[:, None], wave_rest[None, :])  # Shape: (ntime, nwave)
        
        # Get transmission at rest frame wavelengths
        trans = b_rest(wave_rest)
        
        # Convert wavelengths to observer frame for final calculation
        wave_obs = wave_rest * (1.0 + z)
        
        # Convert JAX arrays to numpy for integration
        wave_obs_np = np.array(wave_obs)
        trans_np = np.array(trans)
        rest_flux_np = np.array(rest_flux).squeeze()  # Remove extra dimensions
        
        print("wave_obs_np shape:", wave_obs_np.shape)
        print("trans_np shape:", trans_np.shape)
        print("rest_flux_np shape:", rest_flux_np.shape)
        
        # Compute bandflux for each time using sum with dwave
        # Note: The flux is in rest frame, so we need to scale by a^2
        # We integrate over observer frame wavelengths
        # Note: SNCosmo uses wave * trans * f, where wave is in observer frame
        # and f is in rest frame. We need to scale f by a^2 to convert to observer frame.
        bandflux = np.array([np.sum(wave_obs_np * trans_np * rest_flux_np * a * a) * dwave / HC_ERG_AA])
        bandflux = bandflux.reshape(-1)  # Reshape to 1D array
        print("bandflux shape:", bandflux.shape)
        
        # Apply zeropoint scaling if provided
        if zp is not None:
            ms = get_magsystem(zpsys)
            zpnorm = 10.**(0.4 * zp) / ms.zpbandflux(band)
            bandflux *= zpnorm
            
        return bandflux