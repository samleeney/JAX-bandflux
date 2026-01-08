"""
Source classes for JAX-bandflux providing v3.0 functional API.

This module provides source models for supernova light curve fitting:
- SALT3Source: SALT3-NIR model with stretch and colour parameters
- TimeSeriesSource: Generic spectral time series model (like sncosmo.TimeSeriesSource)

Both use functional API where parameters are passed as dictionaries to methods
rather than stored in the source object.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_supernovae.salt3 import optimized_salt3_multiband_flux, precompute_bandflux_bridge
from jax_supernovae.bandpasses import register_all_bandpasses, get_bandpass
from jax_supernovae.timeseries import (
    timeseries_bandflux,
    timeseries_multiband_flux
)


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
    Basic usage with string band names::

        source = SALT3Source(name='salt3-nir')
        params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}
        flux = source.bandflux(params, 'bessellb', 0.0, zp=27.5, zpsys='ab')

    High-performance usage with precomputed data (for nested sampling)::

        from jax_supernovae.data import load_and_process_data
        times, fluxes, fluxerrs, zps, band_indices, bands, bridges, fixed_z = \
            load_and_process_data('19dwz', fix_z=True)
        source = SALT3Source()
        # Inside likelihood function:
        t0 = 58650.0
        z = fixed_z[0]
        phases = (times - t0) / (1 + z)
        band_names = [bands[i] for i in band_indices]
        fluxes = source.bandflux(params, band_names, phases, zp=zps, zpsys='ab',
                                 band_indices=band_indices, bridges=bridges,
                                 unique_bands=bands)
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
        # Cache of precomputed bridges keyed by band name to avoid slow rebuilding
        self._bridge_cache = {}
        # Cache of compiled bandflux functions keyed by band set and zeropoint usage
        self._compiled_cache = {}

    @property
    def param_names(self):
        """List of SALT3 model parameter names for the functional API.

        Returns the core SALT3 parameters that are passed via the params dict
        in the v3.0 functional API. Note that z and t0 are handled externally
        via phase calculations (phase = (time - t0) / (1 + z)).
        """
        return ['x0', 'x1', 'c']

    def minphase(self):
        """Minimum phase for which the model is defined."""
        return -20.0

    def maxphase(self):
        """Maximum phase for which the model is defined."""
        return 50.0

    def minwave(self):
        """Minimum wavelength for which the model is defined (Angstroms)."""
        return 2000.0

    def maxwave(self):
        """Maximum wavelength for which the model is defined (Angstroms)."""
        return 20000.0

    def __str__(self):
        """String representation."""
        return f"SALT3Source(name='{self.name}', v3.0 functional API)"

    def __repr__(self):
        """Official string representation."""
        return f"SALT3Source(name='{self.name}')"

    def _get_bridges(self, unique_bands):
        """Return cached bridges for the given bands, computing missing ones."""
        bridges = []
        for band in unique_bands:
            if band not in self._bridge_cache:
                self._bridge_cache[band] = precompute_bandflux_bridge(get_bandpass(band))
            bridges.append(self._bridge_cache[band])
        return tuple(bridges)

    def _get_compiled_bandflux(self, band_key, bridges, zpsys, apply_zp):
        """Return a jitted bandflux function for a fixed band set."""
        cache_key = (band_key, zpsys, apply_zp)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        if zpsys not in (None, 'ab'):
            raise ValueError(f"Unsupported magnitude system: {zpsys}")

        band_zp_denoms = jnp.array([bridge['zpbandflux_ab'] for bridge in bridges])

        def _fn(phases, band_indices, params, zps, shifts):
            flux_matrix = optimized_salt3_multiband_flux(
                phases, bridges, params, zps=None, zpsys=zpsys, shifts=shifts
            )
            gathered = flux_matrix[jnp.arange(len(phases)), band_indices]

            if apply_zp:
                zp_norms = 10 ** (0.4 * zps)
                gathered = gathered * (zp_norms / band_zp_denoms[band_indices])
            return gathered

        compiled = jax.jit(_fn)
        self._compiled_cache[cache_key] = compiled
        return compiled

    def bandflux(self, params, bands, phases, zp=None, zpsys=None,
                 band_indices=None, bridges=None, unique_bands=None, shifts=None):
        """Calculate bandflux using the optimized multiband path only.

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
            If `band_indices`/`unique_bands` are provided, `bands` can be None.
        phases : float or array
            Rest-frame phase(s) relative to t0.
        zp : float or array, optional
            Zero point(s) for flux scaling (per observation).
        zpsys : str, optional
            Zero point system ('ab' currently supported).
        band_indices : array, optional
            Indices into unique_bands/bridges arrays (for high-performance path).
        bridges : tuple, optional
            Precomputed bridge data structures keyed to unique_bands.
        unique_bands : list, optional
            List of unique band names corresponding to bridges.
        shifts : array or list, optional
            Wavelength shifts in Angstroms for each unique band.

        Returns
        -------
        flux : array
            Bandflux value(s) with shape matching the requested bands/phases
        """
        if 'x0' not in params or 'x1' not in params or 'c' not in params:
            raise ValueError("params must contain 'x0', 'x1', and 'c'")

        scalar_phase_input = np.isscalar(phases)
        scalar_band_input = isinstance(bands, str)
        scalar_input = scalar_phase_input and (scalar_band_input or bands is None)

        full_params = {
            'z': params.get('z', 0.0),
            't0': params.get('t0', 0.0),
            'x0': params['x0'],
            'x1': params['x1'],
            'c': params['c']
        }

        phases_arr = jnp.atleast_1d(jnp.array(phases))

        # Resolve band metadata
        if band_indices is not None and unique_bands is not None:
            unique_bands_list = list(unique_bands)
            band_indices_arr = jnp.array(band_indices)
        else:
            if bands is None:
                raise ValueError("bands must be provided when band_indices are not supplied")
            bands_arr = np.atleast_1d(np.array(bands))
            unique_bands_list = []
            band_index_list = []
            for band in bands_arr:
                if band not in unique_bands_list:
                    unique_bands_list.append(band)
                band_index_list.append(unique_bands_list.index(band))
            band_indices_arr = jnp.array(band_index_list)

        # Bridges: use provided or cached/computed
        bridges_to_use = tuple(bridges) if bridges is not None else self._get_bridges(unique_bands_list)

        # Align phases and band indices
        phase_len = len(phases_arr)
        band_len = len(band_indices_arr)
        if phase_len == band_len:
            phases_eval = phases_arr
            band_indices_eval = band_indices_arr
        elif phase_len == 1:
            phases_eval = jnp.full(band_len, phases_arr[0])
            band_indices_eval = band_indices_arr
        elif band_len == 1:
            phases_eval = phases_arr
            band_indices_eval = jnp.full(phase_len, int(band_indices_arr[0]))
        else:
            raise ValueError(
                f"Incompatible shapes: bands ({band_len}) and phases ({phase_len}). "
                "Either must be same length, or one must be length 1."
            )

        # Handle zeropoints per observation
        if zp is not None and zpsys is None:
            raise ValueError("zpsys must be provided when zp is specified")
        zps_arr = None
        if zp is not None:
            zps_arr = jnp.atleast_1d(jnp.array(zp))
            if len(zps_arr) == 1:
                zps_arr = jnp.full(len(phases_eval), zps_arr[0])
            elif len(zps_arr) != len(phases_eval):
                raise ValueError(f"zp length ({len(zps_arr)}) must match phases length ({len(phases_eval)})")

        # Handle wavelength shifts per unique band
        shifts_per_band = None
        if shifts is not None:
            shifts_arr = np.atleast_1d(np.array(shifts))
            if len(shifts_arr) == 1:
                shifts_per_band = [float(shifts_arr[0])] * len(bridges_to_use)
            elif len(shifts_arr) == len(bridges_to_use):
                shifts_per_band = [float(s) for s in shifts_arr]
            else:
                raise ValueError(f"shifts length ({len(shifts_arr)}) must match unique bands ({len(bridges_to_use)})")

        band_key = tuple(unique_bands_list)
        apply_zp = zps_arr is not None
        if zps_arr is None:
            zps_arr = jnp.zeros(len(phases_eval))

        if shifts_per_band is None:
            shifts_array = jnp.zeros(len(bridges_to_use))
        else:
            shifts_array = jnp.array(shifts_per_band)

        # Canonicalize zpsys for caching/static use
        zpsys_key = zpsys
        if isinstance(zpsys, (list, tuple, np.ndarray)):
            if len(zpsys) == 0:
                zpsys_key = None
            elif all(z == zpsys[0] for z in zpsys):
                zpsys_key = zpsys[0]
            else:
                raise ValueError("Array-valued zpsys with mixed entries is not supported")

        compiled_fn = self._get_compiled_bandflux(band_key, bridges_to_use, zpsys_key, apply_zp)
        gathered_flux = compiled_fn(phases_eval, band_indices_eval, full_params, zps_arr, shifts_array)

        if scalar_input:
            return gathered_flux[0]
        return gathered_flux

    def bandflux_batch(self, params, bands, phases, zp=None, zpsys=None,
                       band_indices=None, bridges=None, unique_bands=None, shifts=None):
        """Batched bandflux evaluation over multiple parameter sets.

        All core params (x0, x1, c) and optional z, t0 must be 1D arrays of the same length.
        """
        required = ['x0', 'x1', 'c']
        for k in required:
            if k not in params:
                raise ValueError(f"params must contain '{k}' for batched evaluation")

        def _as_1d(name, default):
            val = params.get(name, default)
            arr = jnp.atleast_1d(jnp.asarray(val))
            if arr.ndim != 1:
                raise ValueError(f"Parameter '{name}' must be 1D for batched evaluation")
            return arr

        x0_arr = _as_1d('x0', None)
        x1_arr = _as_1d('x1', None)
        c_arr = _as_1d('c', None)
        z_arr = _as_1d('z', 0.0)
        t0_arr = _as_1d('t0', 0.0)

        batch_size = x0_arr.shape[0]
        for name, arr in [('x1', x1_arr), ('c', c_arr), ('z', z_arr), ('t0', t0_arr)]:
            if arr.shape[0] != batch_size:
                raise ValueError(f"Parameter '{name}' batch size {arr.shape[0]} != {batch_size}")

        def single(param_vec):
            p = {
                'x0': param_vec[0],
                'x1': param_vec[1],
                'c': param_vec[2],
                'z': param_vec[3],
                't0': param_vec[4],
            }
            return self.bandflux(
                p, bands, phases, zp=zp, zpsys=zpsys,
                band_indices=band_indices, bridges=bridges,
                unique_bands=unique_bands, shifts=shifts
            )

        batched_fn = jax.vmap(single)
        params_stack = jnp.stack([x0_arr, x1_arr, c_arr, z_arr, t0_arr], axis=1)
        return batched_fn(params_stack)

    def bandmag(self, params, bands, magsys, phases, band_indices=None,
                bridges=None, unique_bands=None):
        """Calculate magnitude using v3.0 functional API.

        Parameters
        ----------
        params : dict
            Model parameters (x0, x1, c, optionally z and t0)
        bands : str or array-like
            Bandpass name(s)
        magsys : str
            Magnitude system ('ab' or 'vega')
        phases : float or array
            Rest-frame phase(s)
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
        if magsys == 'ab':
            zp = 0.0  # AB magnitudes defined such that zp=0 gives flux in standard units
        else:
            zp = 0.0  # For now, treat all systems the same way

        flux = self.bandflux(params, bands, phases, zp=zp, zpsys=magsys,
                            band_indices=band_indices, bridges=bridges,
                            unique_bands=unique_bands)

        # Convert to magnitude: m = -2.5 * log10(flux)
        # Avoid log of zero/negative values
        flux_safe = jnp.where(flux > 0, flux, jnp.nan)
        mag = -2.5 * jnp.log10(flux_safe)

        return mag


class TimeSeriesSource:
    """JAX implementation of custom SED time series source.

    Matches sncosmo.TimeSeriesSource API with functional parameter passing.
    Enables fitting arbitrary spectral time series models on GPU with JAX.

    This class provides a flexible interface for fitting custom supernova models
    defined by a 2D grid of flux values across phase and wavelength. It uses
    bicubic interpolation (matching sncosmo) and supports both simple usage and
    high-performance modes for MCMC/nested sampling.

    Parameters
    ----------
    phase : array_like
        1D array of phase values (days) defining the model grid.
        Must be sorted in ascending order. Shape (n_phase,)
    wave : array_like
        1D array of wavelength values (Angstroms) defining the model grid.
        Must be sorted in ascending order. Shape (n_wave,)
    flux : array_like
        2D array of flux values (erg/s/cm²/Å) with shape (n_phase, n_wave).
        flux[i, j] is the flux at phase[i] and wavelength wave[j].
    zero_before : bool, optional
        If True, flux is zero for phases before minphase. If False,
        extrapolates using edge values. Default is False.
    time_spline_degree : int, optional
        Degree of interpolation in time direction. 1 for linear, 3 for cubic.
        Default is 3 (matches sncosmo default).
    name : str, optional
        Name for this source model.
    version : str, optional
        Version identifier for this source model.

    Examples
    --------
    Basic usage::

        import numpy as np
        from jax_supernovae import TimeSeriesSource

        # Create simple Gaussian model
        phase = np.linspace(-20, 50, 100)
        wave = np.linspace(3000, 9000, 200)
        # Gaussian in time and wavelength
        p_grid, w_grid = np.meshgrid(phase, wave, indexing='ij')
        flux = np.exp(-0.5 * (p_grid/10)**2) * np.exp(-0.5 * ((w_grid-5000)/1000)**2)
        flux *= 1e-15  # Scale to realistic flux levels

        source = TimeSeriesSource(phase, wave, flux)

        # Calculate bandflux (functional API)
        params = {'amplitude': 1.0}
        flux_b = source.bandflux(params, 'bessellb', 0.0, zp=25.0, zpsys='ab')

    Notes
    -----
    - Uses functional API: parameters passed to methods, not stored
    - Compatible with JAX JIT compilation and GPU acceleration
    - Bicubic interpolation in 2D (phase and wavelength)
    - Matches sncosmo numerical results to ~0.01%
    """

    _param_names = ['amplitude']

    def __init__(self, phase, wave, flux, zero_before=False,
                 time_spline_degree=3, name=None, version=None):
        """Initialise TimeSeriesSource."""
        # Convert to numpy for validation
        phase = np.asarray(phase)
        wave = np.asarray(wave)
        flux = np.asarray(flux)

        # Validate inputs
        if phase.ndim != 1:
            raise ValueError(f"phase must be 1D array, got shape {phase.shape}")
        if wave.ndim != 1:
            raise ValueError(f"wave must be 1D array, got shape {wave.shape}")
        if flux.ndim != 2:
            raise ValueError(f"flux must be 2D array, got shape {flux.shape}")
        if flux.shape != (len(phase), len(wave)):
            raise ValueError(
                f"flux shape {flux.shape} must match (len(phase), len(wave)) = "
                f"({len(phase)}, {len(wave)})"
            )

        # Check grids are sorted
        if not np.all(np.diff(phase) > 0):
            raise ValueError("phase grid must be sorted in ascending order")
        if not np.all(np.diff(wave) > 0):
            raise ValueError("wave grid must be sorted in ascending order")

        # Validate time_spline_degree
        if time_spline_degree not in [1, 3]:
            raise ValueError(
                f"time_spline_degree must be 1 (linear) or 3 (cubic), "
                f"got {time_spline_degree}"
            )

        # Store metadata
        self.name = name
        self.version = version
        self._zero_before = zero_before
        self._time_degree = time_spline_degree

        # Convert to JAX arrays (float64 for precision)
        self._phase = jnp.array(phase, dtype=jnp.float64)
        self._wave = jnp.array(wave, dtype=jnp.float64)
        self._flux = jnp.array(flux, dtype=jnp.float64)

        # Cache bounds for quick access
        self._minphase = float(phase[0])
        self._maxphase = float(phase[-1])
        self._minwave = float(wave[0])
        self._maxwave = float(wave[-1])

        # Register all bandpasses to ensure they're available
        register_all_bandpasses()

    @property
    def param_names(self):
        """List of model parameter names."""
        return self._param_names

    def minphase(self):
        """Minimum phase of model."""
        return self._minphase

    def maxphase(self):
        """Maximum phase of model."""
        return self._maxphase

    def minwave(self):
        """Minimum wavelength of model."""
        return self._minwave

    def maxwave(self):
        """Maximum wavelength of model."""
        return self._maxwave

    def __str__(self):
        """String representation."""
        name_str = f"'{self.name}'" if self.name else 'unnamed'
        return (f"TimeSeriesSource({name_str}, "
                f"phase=[{self._minphase:.1f}, {self._maxphase:.1f}] days, "
                f"wave=[{self._minwave:.0f}, {self._maxwave:.0f}] Å)")

    def __repr__(self):
        """Official string representation."""
        return (f"TimeSeriesSource(name={self.name!r}, version={self.version!r}, "
                f"zero_before={self._zero_before}, time_spline_degree={self._time_degree})")

    def bandflux(self, params, bands, phases, zp=None, zpsys=None,
                 band_indices=None, bridges=None, unique_bands=None):
        """Calculate bandflux using functional API.

        Parameters
        ----------
        params : dict
            Parameter dictionary. Must contain 'amplitude'.
        bands : str, list, or None
            Bandpass name(s). Use None in optimised mode with bridges.
        phases : float or array_like
            Rest-frame phase(s) at which to evaluate flux (days).
        zp : float or array_like, optional
            Zero point(s). If provided, zpsys must also be given.
        zpsys : str, optional
            Zero point system (e.g., 'ab'). Required if zp is provided.
        band_indices : array_like, optional
            (Optimised mode) Integer indices mapping observations to unique_bands.
        bridges : tuple of dict, optional
            (Optimised mode) Pre-computed bandpass bridges.
        unique_bands : list, optional
            (Optimised mode) List of unique band names corresponding to bridges.

        Returns
        -------
        float or jnp.array
            Bandflux value(s). Shape matches input phases.
        """
        # Validate params
        if 'amplitude' not in params:
            raise ValueError("params must contain 'amplitude'")

        # Validate zp/zpsys consistency
        if zp is not None and zpsys is None:
            raise ValueError('zpsys must be given if zp is not None')

        # Extract amplitude
        amplitude = params['amplitude']

        # Check if input is scalar
        scalar_phase_input = np.isscalar(phases)
        scalar_band_input = isinstance(bands, str)
        scalar_input = scalar_phase_input and scalar_band_input

        # Convert phases to JAX array
        phases = jnp.atleast_1d(jnp.array(phases))

        # Handle zp
        if zp is not None:
            zps = jnp.atleast_1d(jnp.array(zp))
        else:
            zps = None

        # High-performance path: use precomputed bridges
        if bridges is not None and band_indices is not None and unique_bands is not None:
            band_indices_arr = jnp.array(band_indices, dtype=jnp.int32)

            # Ensure zps has the right length if provided
            if zps is not None:
                if len(zps) == 1:
                    zps = jnp.full(len(phases), zps[0])
                elif len(zps) != len(phases):
                    raise ValueError(
                        f"zp length ({len(zps)}) must match phases length ({len(phases)})"
                    )

            # Calculate model fluxes using optimised multiband function
            model_fluxes = timeseries_multiband_flux(
                phases, bridges, band_indices_arr,
                self._phase, self._wave, self._flux,
                amplitude, self._zero_before, self._minphase, self._time_degree,
                zps=zps, zpsys=zpsys
            )

            # Return scalar if input was scalar
            if scalar_input:
                return model_fluxes[0]
            return model_fluxes

        # Standard path: create bridges on the fly
        if isinstance(bands, str):
            bands_arr = [bands]
        else:
            bands_arr = list(bands) if not isinstance(bands, list) else bands

        phases_arr = jnp.atleast_1d(jnp.array(phases))

        # Handle zp array
        if zps is not None:
            if len(zps) == 1:
                zps_arr = jnp.full(len(phases_arr), zps[0])
            elif len(zps) != len(phases_arr):
                raise ValueError(
                    f"zp length ({len(zps)}) must match phases length ({len(phases_arr)})"
                )
            else:
                zps_arr = zps
        else:
            zps_arr = None

        # If phases and bands have same length
        if len(bands_arr) == len(phases_arr):
            fluxes = []
            for i, (band, phase) in enumerate(zip(bands_arr, phases_arr)):
                bandpass = get_bandpass(band)
                bridge = precompute_bandflux_bridge(bandpass)
                curr_zp = zps_arr[i] if zps_arr is not None else None
                flux = timeseries_bandflux(
                    phase, bridge, self._phase, self._wave, self._flux,
                    amplitude, self._zero_before, self._minphase, self._time_degree,
                    zp=curr_zp, zpsys=zpsys
                )
                fluxes.append(flux)
            result = jnp.stack(fluxes)
            return result[0] if scalar_input else result

        # If single band, multiple phases
        elif len(bands_arr) == 1:
            bandpass = get_bandpass(bands_arr[0])
            bridge = precompute_bandflux_bridge(bandpass)
            fluxes = []
            for i, phase in enumerate(phases_arr):
                curr_zp = zps_arr[i] if zps_arr is not None else None
                flux = timeseries_bandflux(
                    phase, bridge, self._phase, self._wave, self._flux,
                    amplitude, self._zero_before, self._minphase, self._time_degree,
                    zp=curr_zp, zpsys=zpsys
                )
                fluxes.append(flux)
            result = jnp.stack(fluxes)
            return result[0] if scalar_input else result

        # If single phase, multiple bands
        elif len(phases_arr) == 1:
            phase = phases_arr[0]
            fluxes = []
            for i, band in enumerate(bands_arr):
                bandpass = get_bandpass(band)
                bridge = precompute_bandflux_bridge(bandpass)
                curr_zp = zps_arr[i] if zps_arr is not None else None
                flux = timeseries_bandflux(
                    phase, bridge, self._phase, self._wave, self._flux,
                    amplitude, self._zero_before, self._minphase, self._time_degree,
                    zp=curr_zp, zpsys=zpsys
                )
                fluxes.append(flux)
            return jnp.stack(fluxes)

        else:
            raise ValueError(
                f"Incompatible shapes: bands ({len(bands_arr)}) and phases ({len(phases_arr)}). "
                "Either must be same length, or one must be length 1."
            )

    def bandmag(self, params, bands, magsys, phases, band_indices=None,
                bridges=None, unique_bands=None):
        """Calculate magnitude using functional API.

        Parameters
        ----------
        params : dict
            Model parameters. Must contain 'amplitude'.
        bands : str or array-like
            Bandpass name(s)
        magsys : str
            Magnitude system (e.g., 'ab')
        phases : float or array
            Rest-frame phase(s)
        band_indices, bridges, unique_bands : optional
            For high-performance mode

        Returns
        -------
        mag : float or array
            Magnitude value(s). Returns NaN for flux ≤ 0.
        """
        # Get flux at appropriate zero point for magnitude system
        if magsys == 'ab':
            zp = 0.0
        else:
            zp = 0.0

        flux = self.bandflux(params, bands, phases, zp=zp, zpsys=magsys,
                            band_indices=band_indices, bridges=bridges,
                            unique_bands=unique_bands)

        # Convert to magnitude: m = -2.5 * log10(flux)
        flux_safe = jnp.where(flux > 0, flux, jnp.nan)
        mag = -2.5 * jnp.log10(flux_safe)

        return mag
