import jax.numpy as jnp
from .core import HC_ERG_AA, C_AA_PER_S, trapz
import sncosmo

class Model:
    def __init__(self, source=None):
        self.source = source
        self._parameters = {}
        self.wave = None
        self.flux = None
        
    @property
    def parameters(self):
        return self._parameters
        
    @parameters.setter
    def parameters(self, params):
        self._parameters = params
        
    def add_param(self, name, value):
        """Add a parameter to the model."""
        if name in self._parameters:
            raise ValueError(f"Parameter {name} already exists")
        self._parameters[name] = value
        
    def _compute_bandflux(self, flux, wave_obs, band):
        """Compute band flux in photons/s/cm^2."""
        # Get bandpass transmission
        trans = band(wave_obs)
        
        # Convert from erg/s/cm^2/A to photons/s/cm^2
        photon_flux = flux * wave_obs * trans / HC_ERG_AA
        
        # Integrate over wavelength
        return trapz(photon_flux, wave_obs, axis=1)
        
    def bandflux(self, band, time, zp=None, zpsys=None):
        """Calculate flux through the bandpass(es) in photons/s/cm^2.
        
        Parameters
        ----------
        band : str or Bandpass
            Bandpass or name of bandpass.
        time : float or list_like
            Time(s) at which to evaluate flux.
        zp : float or list_like, optional
            If given, zeropoint to normalize the flux to.
        zpsys : str or list_like, optional
            Name of magnitude system that zp is in.
            
        Returns
        -------
        flux : float or ndarray
            Flux in photons/s/cm^2.
        """
        # Convert inputs to JAX arrays
        time = jnp.asarray(time)
        time = time.reshape(-1)  # Ensure 1D
        
        # Get bandpass object if string is given
        if isinstance(band, str):
            band = sncosmo.get_bandpass(band)
            
        # Get wavelength array
        wave = jnp.array(self.wave)
        
        # Apply redshift
        redshift = self._parameters.get('z', 0.0)
        wave_obs = wave * (1 + redshift)
        
        # Evaluate the model flux at given times and wavelengths
        flux = self.flux(time[:, None], wave_obs[None, :])  # Shape: (ntimes, nwave)
        
        # Compute integrated flux
        integrated_flux = self._compute_bandflux(flux, wave_obs, band)
        
        # Apply zeropoint if requested
        if zp is not None:
            if zpsys is None:
                raise ValueError("zpsys must be given if zp is not None")
            zp_flux = sncosmo.get_magsystem(zpsys).zpbandflux(band)
            integrated_flux = integrated_flux / zp_flux * 10**(-0.4 * (zp - 25))
            
        return integrated_flux 