"""TimeSeriesSource implementation for JAX-bandflux.

This module provides JAX-compatible functions for fitting arbitrary spectral
time series models. It implements bicubic interpolation matching sncosmo's
TimeSeriesSource while being fully compatible with JAX JIT compilation and
GPU acceleration.

Key functions:
- interpolate_timeseries_2d: 2D interpolation with selectable degree
- timeseries_flux: Flux calculation with amplitude scaling and zero_before
- timeseries_bandflux: Single-band bandflux calculation
- timeseries_multiband_flux: Vectorised multi-band calculation for fitting

All functions use JAX primitives and are JIT-compilable.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from jax import vmap
from jax_supernovae.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from jax_supernovae.utils import bandflux_integration
from jax_supernovae.salt3 import kernval, find_index, compute_interpolation_weights

# Enable float64 precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def interpolate_timeseries_2d(phase, wave, phase_grid, wave_grid, flux_grid,
                               time_degree):
    """2D interpolation for timeseries flux grids.

    Performs 2D interpolation on a regular grid with selectable interpolation
    degree in the time direction. Uses bicubic (degree=3) or bilinear (degree=1)
    interpolation to match sncosmo's TimeSeriesSource behaviour.

    Parameters
    ----------
    phase : float
        Phase value to interpolate at (rest-frame days)
    wave : float
        Wavelength value to interpolate at (Angstroms)
    phase_grid : jnp.array
        1D array of phase values defining the model grid (sorted ascending)
    wave_grid : jnp.array
        1D array of wavelength values defining the model grid (sorted ascending)
    flux_grid : jnp.array
        2D array of flux values with shape (len(phase_grid), len(wave_grid))
        Units: erg/s/cm²/Å
    time_degree : int
        Interpolation degree in time direction: 1 for linear, 3 for cubic

    Returns
    -------
    float
        Interpolated flux value

    Notes
    -----
    - Extrapolates using edge values when outside grid bounds
    - Uses bilinear interpolation near grid boundaries for stability
    - Bicubic interpolation uses same kernel as SALT3 (a=-0.5)
    - Wavelength direction always uses cubic interpolation (degree=3)
    - zero_before handling is done in timeseries_flux(), not here

    Implementation follows SALT3's interpolate_2d pattern but with selectable
    time interpolation degree to match sncosmo's time_spline_degree parameter.

    Examples
    --------
    >>> phase_grid = jnp.linspace(-20, 50, 100)
    >>> wave_grid = jnp.linspace(3000, 9000, 200)
    >>> flux_grid = jnp.ones((100, 200)) * 1e-15
    >>> flux = interpolate_timeseries_2d(0.0, 5000.0, phase_grid, wave_grid,
    ...                                  flux_grid, time_degree=3)
    """
    # Compute weights for both dimensions
    ix, dx, x_in_bounds, x_near_boundary = compute_interpolation_weights(
        phase, phase_grid
    )
    iy, dy, y_in_bounds, y_near_boundary = compute_interpolation_weights(
        wave, wave_grid
    )

    # Clamp dx and dy to [0, 1] for proper extrapolation
    # When out of bounds, we want to use edge values, which means:
    # - If below grid: use dx=0 (lower edge)
    # - If above grid: use dx=1 (upper edge)
    dx = jnp.clip(dx, 0.0, 1.0)
    dy = jnp.clip(dy, 0.0, 1.0)

    # Check if we're in bounds
    in_bounds = x_in_bounds & y_in_bounds

    # Determine if we should use linear interpolation
    # Use linear if: (1) out of bounds, (2) near boundary, or (3) time_degree=1
    use_linear = (~in_bounds) | x_near_boundary | y_near_boundary | (time_degree == 1)

    # Get corner values for bilinear interpolation
    z00 = flux_grid[ix, iy]
    z01 = flux_grid[ix, iy + 1]
    z10 = flux_grid[ix + 1, iy]
    z11 = flux_grid[ix + 1, iy + 1]

    # Bilinear interpolation formula
    linear_result = (
        z00 * (1 - dx) * (1 - dy) +
        z10 * dx * (1 - dy) +
        z01 * (1 - dx) * dy +
        z11 * dx * dy
    )

    # For bicubic interpolation (time_degree=3), pad the array with edge values
    padded = jnp.pad(flux_grid, ((1, 1), (1, 1)), mode='edge')

    # Get 4x4 grid for bicubic interpolation
    ix_pad = ix + 1  # Adjust for padding
    iy_pad = iy + 1
    grid = lax.dynamic_slice(padded, (ix_pad - 1, iy_pad - 1), (4, 4))

    # Calculate bicubic weights using kernval function (reused from SALT3)
    wx = jnp.array([
        kernval(dx + 1.0),
        kernval(dx),
        kernval(dx - 1.0),
        kernval(dx - 2.0)
    ])

    wy = jnp.array([
        kernval(dy + 1.0),
        kernval(dy),
        kernval(dy - 1.0),
        kernval(dy - 2.0)
    ])

    # Calculate bicubic interpolation result
    cubic_result = jnp.sum(jnp.outer(wx, wy) * grid)

    # Select interpolation result based on conditions
    result = jnp.where(use_linear, linear_result, cubic_result)

    # Note: We do NOT return 0 if out of bounds - extrapolation is handled
    # by zero_before parameter in timeseries_flux(). This allows the function
    # to extrapolate using edge values when zero_before=False.

    return result


@jax.jit
def timeseries_flux(phase, wave, phase_grid, wave_grid, flux_grid,
                    amplitude, zero_before, minphase, time_degree):
    """Calculate timeseries flux with scaling and zero_before handling.

    Computes flux at given phase and wavelength by interpolating from the
    model grid, applying amplitude scaling, and optionally zeroing flux
    before the minimum phase.

    Parameters
    ----------
    phase : float
        Rest-frame phase (days relative to some reference)
    wave : float
        Wavelength (Angstroms)
    phase_grid : jnp.array
        1D array of phase values in model grid
    wave_grid : jnp.array
        1D array of wavelength values in model grid
    flux_grid : jnp.array
        2D flux array with shape (len(phase_grid), len(wave_grid))
        Units: erg/s/cm²/Å
    amplitude : float
        Amplitude scaling parameter
    zero_before : bool
        If True, return 0 for phase < minphase
    minphase : float
        Minimum phase value (used with zero_before)
    time_degree : int
        Time interpolation degree (1=linear, 3=cubic)

    Returns
    -------
    float
        Scaled flux value (erg/s/cm²/Å)

    Notes
    -----
    The flux is calculated as:
        flux = amplitude * interpolate(phase, wave)

    If zero_before is True:
        flux = 0 if phase < minphase, otherwise flux

    This matches sncosmo's TimeSeriesSource._flux() method exactly.

    Examples
    --------
    >>> flux = timeseries_flux(0.0, 5000.0, phase_grid, wave_grid,
    ...                        flux_grid, amplitude=1.0, zero_before=False,
    ...                        minphase=-20.0, time_degree=3)
    """
    # Interpolate flux from grid
    flux_interp = interpolate_timeseries_2d(
        phase, wave, phase_grid, wave_grid, flux_grid, time_degree
    )

    # Scale by amplitude
    flux_scaled = amplitude * flux_interp

    # Apply zero_before: set to 0 if phase < minphase
    # Use jnp.where for JAX compatibility (no boolean indexing)
    flux_final = jnp.where(
        zero_before & (phase < minphase),
        0.0,
        flux_scaled
    )

    return flux_final


# Vectorise timeseries_flux over wavelengths for bandflux integration
# This allows efficient calculation of flux at all integration wavelengths
timeseries_flux_vmap_wave = vmap(
    timeseries_flux,
    in_axes=(None, 0, None, None, None, None, None, None, None)  # vmap over wave (axis 0)
)


@partial(jax.jit, static_argnames=['zpsys'])
def timeseries_bandflux(phase, bridge, phase_grid, wave_grid, flux_grid,
                        amplitude, zero_before, minphase, time_degree,
                        zp=None, zpsys='ab'):
    """Calculate bandflux for timeseries source through single bandpass.

    Computes synthetic photometry by integrating the model spectrum through
    a bandpass transmission curve. Supports optional zero-point scaling.

    Parameters
    ----------
    phase : float or array_like
        Rest-frame phase(s) at which to evaluate flux (days)
    bridge : dict
        Pre-computed bandpass integration grid containing:
        - 'wave': wavelength grid (Angstroms)
        - 'dwave': grid spacing (Angstroms)
        - 'trans': transmission values on wave grid
    phase_grid : jnp.array
        Phase grid for model interpolation
    wave_grid : jnp.array
        Wavelength grid for model interpolation
    flux_grid : jnp.array
        2D flux array (n_phase × n_wave)
    amplitude : float
        Amplitude scaling parameter
    zero_before : bool
        If True, zero flux before minphase
    minphase : float
        Minimum phase of model
    time_degree : int
        Time interpolation degree (1 or 3)
    zp : float, optional
        Zero point magnitude. If None, no scaling applied.
    zpsys : str, optional
        Zero point system ('ab', etc.). Default 'ab'.

    Returns
    -------
    float or jnp.array
        Bandflux (photons/s/cm²). Shape matches input phase.

    Notes
    -----
    Integration formula: ∫ λ T(λ) F(λ) dλ / (hc)

    Uses the shared bandflux_integration function for consistency with
    other source models (SALT3, etc.).

    Examples
    --------
    >>> # Single phase
    >>> flux = timeseries_bandflux(0.0, bridge, phase_grid, wave_grid,
    ...                            flux_grid, amplitude=1.0, zero_before=False,
    ...                            minphase=-20.0, time_degree=3)
    >>>
    >>> # Array of phases
    >>> phases = jnp.array([0, 5, 10, 15])
    >>> fluxes = timeseries_bandflux(phases, bridge, phase_grid, wave_grid,
    ...                              flux_grid, amplitude=1.0, ...)
    """
    # Check scalar vs array input
    is_scalar = jnp.ndim(phase) == 0
    phase = jnp.atleast_1d(phase)

    # Extract integration grid from bridge
    wave = bridge['wave']
    dwave = bridge['dwave']
    trans = bridge['trans']

    # Calculate flux at all integration wavelengths for all phases
    # Shape: (n_phases, n_wave)
    def flux_at_phase(p):
        """Calculate flux at all wavelengths for a single phase."""
        return timeseries_flux_vmap_wave(
            p, wave, phase_grid, wave_grid, flux_grid,
            amplitude, zero_before, minphase, time_degree
        )

    # Vmap over phases
    flux_all = vmap(flux_at_phase)(phase)

    # Integrate using shared integration function
    # bandflux_integration expects flux with shape (..., N_wave)
    result = bandflux_integration(wave, trans, flux_all, dwave)

    # Apply zero point scaling if provided
    if zp is not None:
        if zpsys == 'ab':
            # Calculate zpbandflux for AB system
            # AB spectrum: 3631 × 10^-23 erg/s/cm²/Hz
            # Convert to F_lambda and integrate
            from jax_supernovae.salt3 import H_ERG_S
            zpbandflux = 3631e-23 * dwave / H_ERG_S * jnp.sum(trans / wave)
            zpnorm = 10.0**(0.4 * zp) / zpbandflux
            result = result * zpnorm
        else:
            raise ValueError(f"Unsupported magnitude system: {zpsys}")

    # Return scalar if input was scalar
    if is_scalar:
        result = result[0]

    return result


@partial(jax.jit, static_argnames=['zpsys'])
def timeseries_multiband_flux(phases, bridges, band_indices, phase_grid,
                               wave_grid, flux_grid, amplitude, zero_before,
                               minphase, time_degree, zps=None, zpsys='ab'):
    """Vectorised multi-band bandflux calculation for light curve fitting.

    High-performance function for calculating bandflux across multiple bands
    and phases simultaneously. This is the optimised path used in MCMC/nested
    sampling where the same model is evaluated many times.

    Parameters
    ----------
    phases : jnp.array
        Array of rest-frame phases with shape (N,) for N observations
    bridges : tuple of dict
        Tuple of M pre-computed bandpass bridges for unique bands
    band_indices : jnp.array
        Integer array with shape (N,) mapping each observation to its bridge.
        band_indices[i] gives the index into bridges for observation i.
    phase_grid : jnp.array
        Phase grid for model interpolation
    wave_grid : jnp.array
        Wavelength grid for model interpolation
    flux_grid : jnp.array
        2D flux array (n_phase × n_wave)
    amplitude : float
        Amplitude scaling parameter
    zero_before : bool
        If True, zero flux before minphase
    minphase : float
        Minimum phase of model
    time_degree : int
        Time interpolation degree (1 or 3)
    zps : jnp.array, optional
        Zero points for each observation. Shape (N,).
    zpsys : str, optional
        Zero point system. Default 'ab'.

    Returns
    -------
    jnp.array
        Bandflux values for all observations. Shape (N,).

    Notes
    -----
    This function is designed for maximum performance in fitting workflows:
    - Vectorised over all observations
    - Pre-computed bridges avoid repeated bandpass calculations
    - band_indices enable efficient lookup of correct bridge per observation

    Typical usage in a likelihood function:
    >>> @jax.jit
    >>> def loglikelihood(amplitude):
    >>>     model_fluxes = timeseries_multiband_flux(
    >>>         phases, bridges, band_indices, phase_grid, wave_grid,
    >>>         flux_grid, amplitude, zero_before, minphase, time_degree,
    >>>         zps=zps, zpsys='ab'
    >>>     )
    >>>     chi2 = jnp.sum(((data_fluxes - model_fluxes) / errors)**2)
    >>>     return -0.5 * chi2

    Examples
    --------
    >>> # Simulated light curve with 50 observations in 3 bands
    >>> n_obs = 50
    >>> phases = jnp.linspace(-10, 40, n_obs)
    >>> band_indices = jnp.array([0, 1, 2] * (n_obs // 3) + [0] * (n_obs % 3))
    >>> bridges = (bridge_b, bridge_v, bridge_r)  # 3 unique bands
    >>> zps = jnp.ones(n_obs) * 27.5
    >>>
    >>> fluxes = timeseries_multiband_flux(
    >>>     phases, bridges, band_indices, phase_grid, wave_grid,
    >>>     flux_grid, amplitude=1.0, zero_before=False,
    >>>     minphase=-20.0, time_degree=3, zps=zps, zpsys='ab'
    >>> )
    """
    n_obs = len(phases)

    # Stack bridge arrays for JAX-compatible indexing
    # Note: Bridges may have different wavelength grid sizes, so we need to pad
    # them to a common length before stacking
    max_wave_len = max(len(b['wave']) for b in bridges)

    def pad_array_edge(arr, target_len):
        """Pad array to target length with edge values."""
        if len(arr) == target_len:
            return arr
        else:
            pad_width = target_len - len(arr)
            # Use edge mode for wave (to avoid zeros), constant for trans (zeros are fine)
            return jnp.pad(arr, (0, pad_width), mode='edge')

    def pad_array_zero(arr, target_len):
        """Pad array to target length with zeros."""
        if len(arr) == target_len:
            return arr
        else:
            pad_width = target_len - len(arr)
            return jnp.pad(arr, (0, pad_width), mode='constant', constant_values=0)

    # Pad and stack all bridge components
    # Wave: pad with edge values to avoid zeros (which can cause issues)
    # Trans: pad with zeros (padded region contributes 0 to integration)
    waves_stacked = jnp.stack([pad_array_edge(b['wave'], max_wave_len) for b in bridges])
    dwaves_stacked = jnp.array([b['dwave'] for b in bridges])
    trans_stacked = jnp.stack([pad_array_zero(b['trans'], max_wave_len) for b in bridges])

    # Create function that uses JAX indexing (works with traced values)
    # Note: We use padded arrays directly. Padding trans with zeros means
    # the padded region contributes 0 to the integration, which is correct.
    if zps is None:
        # No zeropoint - simpler path
        def single_observation(phase, band_idx):
            """Calculate bandflux without zeropoint."""
            wave_select = waves_stacked[band_idx]
            dwave_select = dwaves_stacked[band_idx]
            trans_select = trans_stacked[band_idx]

            bridge_select = {
                'wave': wave_select,
                'dwave': dwave_select,
                'trans': trans_select
            }

            return timeseries_bandflux(
                phase, bridge_select, phase_grid, wave_grid, flux_grid,
                amplitude, zero_before, minphase, time_degree,
                zp=None, zpsys=zpsys
            )

        fluxes = vmap(single_observation)(phases, band_indices)

    else:
        # With zeropoint
        def single_observation_with_zp(phase, band_idx, zp_single):
            """Calculate bandflux with zeropoint."""
            wave_select = waves_stacked[band_idx]
            dwave_select = dwaves_stacked[band_idx]
            trans_select = trans_stacked[band_idx]

            bridge_select = {
                'wave': wave_select,
                'dwave': dwave_select,
                'trans': trans_select
            }

            return timeseries_bandflux(
                phase, bridge_select, phase_grid, wave_grid, flux_grid,
                amplitude, zero_before, minphase, time_degree,
                zp=zp_single, zpsys=zpsys
            )

        fluxes = vmap(single_observation_with_zp)(phases, band_indices, zps)

    return fluxes
