"""
SALT3 Source class providing v3.0 functional API.

This module provides a simplified interface to the SALT3-NIR model with a functional
API where parameters are passed as dictionaries to methods rather than stored in
the source object.
"""

import jax.numpy as jnp
import numpy as np
from jax_supernovae.salt3 import optimized_salt3_multiband_flux, optimized_salt3_bandflux, precompute_bandflux_bridge
from jax_supernovae.bandpasses import register_all_bandpasses, get_bandpass


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

    @property
    def param_names(self):
        """List of SALT3 model parameter names.

        Returns all five SALT3 parameters, even though the v3.0 functional API
        allows you to pass only a subset (e.g., just x0, x1, c with z and t0
        handled externally via phases).
        """
        return ['z', 't0', 'x0', 'x1', 'c']

    @property
    def minphase(self):
        """Minimum phase for which the model is defined."""
        return -20.0

    @property
    def maxphase(self):
        """Maximum phase for which the model is defined."""
        return 50.0

    @property
    def minwave(self):
        """Minimum wavelength for which the model is defined (Angstroms)."""
        return 2000.0

    @property
    def maxwave(self):
        """Maximum wavelength for which the model is defined (Angstroms)."""
        return 18000.0

    def __str__(self):
        """String representation."""
        return f"SALT3Source(name='{self.name}')"

    def __repr__(self):
        """Official string representation."""
        return f"SALT3Source(name='{self.name}')"

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

        # Standard path: create bridges on the fly
        # This is simpler but slower - fine for one-off calculations
        # but inefficient for nested sampling (use bridges parameter instead)

        # Determine if input is scalar
        scalar_input = isinstance(bands, str) and np.isscalar(phases)

        # Convert bands and phases to arrays
        if isinstance(bands, str):
            bands_arr = [bands]
        else:
            bands_arr = list(bands) if not isinstance(bands, list) else bands

        phases_arr = jnp.atleast_1d(jnp.array(phases))

        # Handle zp
        if zps is not None:
            if len(zps) == 1:
                zps_arr = jnp.full(len(phases_arr), zps[0])
            elif len(zps) != len(phases_arr):
                raise ValueError(f"zp length ({len(zps)}) must match phases length ({len(phases_arr)})")
            else:
                zps_arr = zps
        else:
            zps_arr = jnp.zeros(len(phases_arr))

        # If phases and bands have same length, calculate one flux per (phase, band) pair
        if len(bands_arr) == len(phases_arr):
            fluxes = []
            for i, (band, phase) in enumerate(zip(bands_arr, phases_arr)):
                bandpass = get_bandpass(band)
                bridge = precompute_bandflux_bridge(bandpass)
                flux = optimized_salt3_bandflux(
                    phase, bridge['wave'], bridge['dwave'], bridge['trans'],
                    full_params, zp=zps_arr[i], zpsys=zpsys
                )
                fluxes.append(flux)
            result = jnp.array(fluxes)
            return result[0] if scalar_input else result

        # If single band, multiple phases
        elif len(bands_arr) == 1:
            bandpass = get_bandpass(bands_arr[0])
            bridge = precompute_bandflux_bridge(bandpass)
            fluxes = []
            for i, phase in enumerate(phases_arr):
                flux = optimized_salt3_bandflux(
                    phase, bridge['wave'], bridge['dwave'], bridge['trans'],
                    full_params, zp=zps_arr[i], zpsys=zpsys
                )
                fluxes.append(flux)
            result = jnp.array(fluxes)
            return result[0] if scalar_input else result

        # If single phase, multiple bands
        elif len(phases_arr) == 1:
            phase = phases_arr[0]
            fluxes = []
            for i, band in enumerate(bands_arr):
                bandpass = get_bandpass(band)
                bridge = precompute_bandflux_bridge(bandpass)
                flux = optimized_salt3_bandflux(
                    phase, bridge['wave'], bridge['dwave'], bridge['trans'],
                    full_params, zp=zps_arr[i], zpsys=zpsys
                )
                fluxes.append(flux)
            return jnp.array(fluxes)

        else:
            raise ValueError(
                f"Incompatible shapes: bands ({len(bands_arr)}) and phases ({len(phases_arr)}). "
                "Either must be same length, or one must be length 1."
            )

    def bandmag(self, params, bands, phases, zpsys='ab', band_indices=None,
                bridges=None, unique_bands=None):
        """Calculate magnitude using v3.0 functional API.

        Parameters
        ----------
        params : dict
            Model parameters (x0, x1, c, optionally z and t0)
        bands : str or array-like
            Bandpass name(s)
        phases : float or array
            Rest-frame phase(s)
        zpsys : str, optional
            Zero point system (default: 'ab')
        band_indices : array, optional
            For performance: indices into unique_bands/bridges arrays
        bridges : tuple, optional
            For performance: precomputed bridge data structures
        unique_bands : list, optional
            For performance: list of unique band names

        Returns
        -------
        mag : float or array
            Magnitude value(s)

        Notes
        -----
        Magnitude is calculated as -2.5 * log10(flux/zp0)
        """
        # Get flux at zeropoint
        if zpsys == 'ab':
            zp = 0.0  # AB magnitudes defined such that zp=0 gives flux in standard units
        else:
            zp = 0.0  # For now, treat all systems the same way

        flux = self.bandflux(params, bands, phases, zp=zp, zpsys=zpsys,
                            band_indices=band_indices, bridges=bridges,
                            unique_bands=unique_bands)

        # Convert to magnitude: m = -2.5 * log10(flux)
        # Avoid log of zero/negative values
        flux_safe = jnp.where(flux > 0, flux, jnp.nan)
        mag = -2.5 * jnp.log10(flux_safe)

        return mag
