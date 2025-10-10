"""
SALT3 Source class providing v3.0 functional API.

This module provides a simplified interface to the SALT3-NIR model with a functional
API where parameters are passed as dictionaries to methods rather than stored in
the source object.
"""

import jax.numpy as jnp
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.bandpasses import get_bandpass, register_all_bandpasses


class SALT3Source:
    """SALT3-NIR supernova source model with functional API.

    This class provides a v3.0 functional API where model parameters are passed
    as arguments to methods rather than being stored as instance attributes.

    Parameters
    ----------
    name : str, optional
        Model name (default: 'salt3-nir')

    Examples
    --------
    >>> source = SALT3Source(name='salt3-nir')
    >>> params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
    >>> bands = ['g', 'r', 'i', 'z']
    >>> phases = jnp.array([-10.0, 0.0, 10.0, 20.0])
    >>> fluxes = source.bandflux(params, bands, phases, zp=27.5, zpsys='ab')
    """

    def __init__(self, name='salt3-nir'):
        """Initialize SALT3 source.

        Parameters
        ----------
        name : str
            Model name (currently only 'salt3-nir' is supported)
        """
        if name != 'salt3-nir':
            raise ValueError(f"Only 'salt3-nir' model is supported, got '{name}'")
        self.name = name

        # Register all bandpasses to ensure they're available
        register_all_bandpasses()

    def bandflux(self, params, bands, phases, zp=None, zpsys=None):
        """Calculate bandflux using v3.0 functional API.

        Parameters
        ----------
        params : dict
            Model parameters. Must contain:
            - 'x0': float - Amplitude parameter
            - 'x1': float - Stretch parameter
            - 'c': float - Color parameter
            May optionally contain:
            - 'z': float - Redshift (default: 0.0)
            - 't0': float - Time of peak brightness (default: 0.0)
        bands : str or array-like
            Bandpass name(s). Can be a string for a single band or array of band names.
        phases : float or array
            Rest-frame phase(s) relative to t0
        zp : float or array, optional
            Zero point(s) for flux scaling
        zpsys : str, optional
            Zero point system ('ab' or 'vega')

        Returns
        -------
        flux : array
            Bandflux value(s) with shape matching the broadcast shape of inputs

        Examples
        --------
        Single band and phase:
        >>> source = SALT3Source()
        >>> params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
        >>> flux = source.bandflux(params, 'g', 0.0, zp=27.5, zpsys='ab')

        Multiple bands and phases:
        >>> bands = ['g', 'r', 'i']
        >>> phases = jnp.array([-10.0, 0.0, 10.0])
        >>> fluxes = source.bandflux(params, bands, phases, zp=27.5, zpsys='ab')
        """
        # Ensure params has required keys
        if 'x0' not in params or 'x1' not in params or 'c' not in params:
            raise ValueError("params must contain 'x0', 'x1', and 'c'")

        # Create full parameter dict with defaults
        full_params = {
            'z': params.get('z', 0.0),
            't0': params.get('t0', 0.0),
            'x0': params['x0'],
            'x1': params['x1'],
            'c': params['c']
        }

        # Handle single band case
        if isinstance(bands, str):
            bands = [bands]
            single_band = True
        else:
            single_band = False

        # Get bandpass bridges
        from jax_supernovae.data import create_bridge_for_bands
        bridges = create_bridge_for_bands(bands)

        # Convert phases to JAX array
        phases = jnp.atleast_1d(jnp.array(phases))

        # Handle zp
        if zp is not None:
            zps = jnp.atleast_1d(jnp.array(zp))
        else:
            zps = jnp.zeros(len(phases) * len(bands))

        # Create times array (phases at rest frame)
        # For each phase, we have one observation per band
        times = jnp.repeat(phases, len(bands))

        # Create band_indices array
        band_indices = jnp.tile(jnp.arange(len(bands)), len(phases))

        # Ensure zps has the right length
        if len(zps) == 1:
            zps = jnp.full(len(times), zps[0])
        elif len(zps) == len(phases):
            # Repeat zps for each band
            zps = jnp.repeat(zps, len(bands))

        # Calculate model fluxes
        model_fluxes = optimized_salt3_multiband_flux(
            times, bridges, full_params, zps=zps, zpsys=zpsys
        )

        # Index by band to get final fluxes
        model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]

        # Reshape to (n_phases, n_bands)
        model_fluxes = model_fluxes.reshape(len(phases), len(bands))

        # If single band, squeeze the band dimension
        if single_band:
            model_fluxes = model_fluxes.squeeze(axis=1)

        # If single phase, squeeze the phase dimension
        if len(phases) == 1:
            model_fluxes = model_fluxes.squeeze(axis=0)

        return model_fluxes
