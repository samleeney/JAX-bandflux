"""
SALT3 Source class providing v3.0 functional API.

This module provides a simplified interface to the SALT3-NIR model with a functional
API where parameters are passed as dictionaries to methods rather than stored in
the source object.
"""

import jax.numpy as jnp
import numpy as np
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.bandpasses import register_all_bandpasses


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
    Basic usage with string band names:
    >>> source = SALT3Source(name='salt3-nir')
    >>> params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
    >>> flux = source.bandflux(params, 'bessellb', 0.0, zp=27.5, zpsys='ab')

    High-performance usage with precomputed data (for nested sampling):
    >>> from jax_supernovae.data import load_and_process_data
    >>> times, fluxes, fluxerrs, zps, band_indices, bands, bridges, fixed_z = \\
    ...     load_and_process_data('19dwz', fix_z=True)
    >>> source = SALT3Source()
    >>> # Inside likelihood function:
    >>> t0 = 58650.0
    >>> z = fixed_z[0]
    >>> phases = (times - t0) / (1 + z)
    >>> band_names = [bands[i] for i in band_indices]
    >>> fluxes = source.bandflux(params, band_names, phases, zp=zps, zpsys='ab',
    ...                          band_indices=band_indices, bridges=bridges,
    ...                          unique_bands=bands)
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

    def bandflux(self, params, bands, phases, zp=None, zpsys=None,
                 band_indices=None, bridges=None, unique_bands=None):
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
            Rest-frame phase(s) relative to t0.
        zp : float or array, optional
            Zero point(s) for flux scaling
        zpsys : str, optional
            Zero point system ('ab' or 'vega')
        band_indices : array, optional
            For performance: indices into unique_bands/bridges arrays.
            If provided, must also provide bridges and unique_bands.
        bridges : tuple, optional
            For performance: precomputed bridge data structures.
        unique_bands : list, optional
            For performance: list of unique band names corresponding to bridges.

        Returns
        -------
        flux : array
            Bandflux value(s) with shape matching the input

        Examples
        --------
        Single band and phase:
        >>> source = SALT3Source()
        >>> params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
        >>> flux = source.bandflux(params, 'bessellb', 0.0, zp=27.5, zpsys='ab')

        High-performance mode with precomputed bridges:
        >>> flux = source.bandflux(params, bands, phases, zp=zps, zpsys='ab',
        ...                        band_indices=band_indices, bridges=bridges,
        ...                        unique_bands=unique_bands)
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

        # Convert phases to JAX array
        phases = jnp.atleast_1d(jnp.array(phases))

        # Handle zp
        if zp is not None:
            zps = jnp.atleast_1d(jnp.array(zp))
        else:
            zps = None

        # High-performance path: use precomputed bridges
        if bridges is not None and band_indices is not None and unique_bands is not None:
            # This is the optimized path for nested sampling
            # band_indices and bridges are already set up correctly
            band_indices_arr = jnp.array(band_indices)

            # Ensure zps has the right length
            if zps is not None:
                if len(zps) == 1:
                    zps = jnp.full(len(phases), zps[0])
                elif len(zps) != len(phases):
                    raise ValueError(f"zp length ({len(zps)}) must match phases length ({len(phases)})")
            else:
                zps = jnp.zeros(len(phases))

            # Calculate model fluxes using optimized function
            model_fluxes = optimized_salt3_multiband_flux(
                phases, bridges, full_params, zps=zps, zpsys=zpsys
            )

            # Index by band to get final fluxes
            model_fluxes = model_fluxes[jnp.arange(len(phases)), band_indices_arr]

            # Return scalar if input was scalar
            if len(phases) == 1:
                return model_fluxes[0]
            return model_fluxes

        # Standard path: slower but simpler
        # This creates bridges on the fly - fine for one-off calculations
        # but inefficient for nested sampling
        raise NotImplementedError(
            "SALT3Source currently requires precomputed bridges for performance. "
            "Use load_and_process_data() to get bridges, then pass them to bandflux(). "
            "See examples/ns.py for usage."
        )
