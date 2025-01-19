"""SALT2 model data."""
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import sncosmo

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Get SALT2 source
salt2_source = sncosmo.get_source('salt2')

# Pre-compute the wavelength grid
wave_grid = jnp.array(salt2_source._wave)
phase_grid = jnp.array(salt2_source._phase)

# Pre-compute the template data by evaluating on the grid
# Note: Data is already scaled by 1e-12 in SNCosmo's initialization
# Use numpy arrays with SNCosmo's interpolator
wave_grid_np = np.array(wave_grid)
phase_grid_np = np.array(phase_grid)
M0_data = jnp.array(salt2_source._model['M0'](phase_grid_np, wave_grid_np))
M1_data = jnp.array(salt2_source._model['M1'](phase_grid_np, wave_grid_np))

# Pre-compute color law values on the wavelength grid
# Note: SALT2 color law requires float64 input
wave_grid_f64 = np.array(wave_grid, dtype=np.float64)
colorlaw_grid = jnp.array(salt2_source._colorlaw(wave_grid_f64))

@jax.jit
def _cubic_interp1d_single(x, xp, yp):
    """1D cubic interpolation for a single point in JAX.
    
    Parameters
    ----------
    x : float
        Point at which to evaluate the interpolated value.
    xp : array_like
        The x-coordinates of the data points.
    yp : array_like
        The y-coordinates of the data points, can be 1D or 2D.
    
    Returns
    -------
    float or array_like
        The interpolated value(s).
    """
    # Find index of the interval containing x
    i = jnp.searchsorted(xp, x)
    i = jnp.clip(i, 1, len(xp)-2)  # Need 2 points on each side
    
    # Convert i to int32 for consistent typing
    i = i.astype(jnp.int32)
    
    # Get the 4 surrounding points using dynamic_slice
    x_i = lax.dynamic_slice(xp, (i-1,), (4,))
    
    # Handle both 1D and 2D yp arrays
    if yp.ndim == 1:
        y_i = lax.dynamic_slice(yp, (i-1,), (4,))
    else:
        # For 2D arrays, slice along the second dimension
        # We need to slice each row individually
        def slice_row(row):
            return lax.dynamic_slice(row, (i-1,), (4,))
        y_i = jax.vmap(slice_row)(yp)
    
    # Calculate position within interval
    t = (x - x_i[1]) / (x_i[2] - x_i[1])
    
    # Calculate cubic coefficients
    # Using Catmull-Rom spline formulation
    if yp.ndim == 1:
        c0 = -0.5 * y_i[0] + 1.5 * y_i[1] - 1.5 * y_i[2] + 0.5 * y_i[3]
        c1 = y_i[0] - 2.5 * y_i[1] + 2 * y_i[2] - 0.5 * y_i[3]
        c2 = -0.5 * y_i[0] + 0.5 * y_i[2]
        c3 = y_i[1]
    else:
        c0 = -0.5 * y_i[:, 0] + 1.5 * y_i[:, 1] - 1.5 * y_i[:, 2] + 0.5 * y_i[:, 3]
        c1 = y_i[:, 0] - 2.5 * y_i[:, 1] + 2 * y_i[:, 2] - 0.5 * y_i[:, 3]
        c2 = -0.5 * y_i[:, 0] + 0.5 * y_i[:, 2]
        c3 = y_i[:, 1]
    
    # Evaluate cubic polynomial
    t2 = t * t
    t3 = t2 * t
    return c0 * t3 + c1 * t2 + c2 * t + c3

@jax.jit
def cubic_interp1d(x, xp, yp):
    """1D cubic interpolation in JAX with support for batched inputs.
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the interpolated values.
    xp : array_like
        The x-coordinates of the data points.
    yp : array_like
        The y-coordinates of the data points, can be 1D or 2D.
    
    Returns
    -------
    array_like
        The interpolated values.
    """
    # Reshape x to be 1D if needed
    x_shape = x.shape
    x_flat = x.reshape(-1)
    
    # Use vmap to handle batched inputs
    interp_fn = jax.vmap(_cubic_interp1d_single, in_axes=(0, None, None))
    result = interp_fn(x_flat, xp, yp)
    
    # Reshape result back to original shape
    return result.reshape(x_shape)

@jax.jit
def _salt2_m0_single_wave(phase, wave_idx):
    """Interpolate M0 template at a single wavelength."""
    return cubic_interp1d(phase, phase_grid, M0_data[:, wave_idx])

@jax.jit
def salt2_m0(phase, wave):
    """Interpolate the M0 template at the given phase and wavelength.

    Args:
        phase (float or array-like): The phase(s) at which to evaluate the template.
        wave (float or array-like): The wavelength(s) at which to evaluate the template.

    Returns:
        array-like: The interpolated M0 template values.
    """
    # Get original shapes
    phase_shape = jnp.shape(phase)
    wave_shape = jnp.shape(wave)

    # Reshape inputs to 1D
    phase_flat = jnp.ravel(phase)
    wave_flat = jnp.ravel(wave)

    # First interpolate in phase for each wavelength in the template
    phase_interp = jax.vmap(lambda w: cubic_interp1d(phase_flat, phase_grid, M0_data[:, w]))(jnp.arange(len(wave_grid)))

    # Then interpolate in wavelength for each phase
    result = jax.vmap(lambda p: cubic_interp1d(wave_flat, wave_grid, phase_interp[:, p]))(jnp.arange(len(phase_flat)))

    # Transpose and reshape to match input shapes
    result = result.T

    # Return scalar if both inputs are scalar
    if len(phase_shape) == 0 and len(wave_shape) == 0:
        return float(result)
    
    # Otherwise reshape to broadcast shape
    broadcast_shape = jnp.broadcast_shapes(phase_shape, wave_shape)
    return result.reshape(broadcast_shape)

@jax.jit
def _salt2_m1_single_wave(phase, wave_idx):
    """Interpolate M1 template at a single wavelength."""
    return cubic_interp1d(phase, phase_grid, M1_data[:, wave_idx])

@jax.jit
def salt2_m1(phase, wave):
    """Interpolate the M1 template at the given phase and wavelength.

    Args:
        phase (float or array-like): The phase(s) at which to evaluate the template.
        wave (float or array-like): The wavelength(s) at which to evaluate the template.

    Returns:
        array-like: The interpolated M1 template values.
    """
    # Get original shapes
    phase_shape = jnp.shape(phase)
    wave_shape = jnp.shape(wave)

    # Reshape inputs to 1D
    phase_flat = jnp.ravel(phase)
    wave_flat = jnp.ravel(wave)

    # First interpolate in phase for each wavelength in the template
    phase_interp = jax.vmap(lambda w: cubic_interp1d(phase_flat, phase_grid, M1_data[:, w]))(jnp.arange(len(wave_grid)))

    # Then interpolate in wavelength for each phase
    result = jax.vmap(lambda p: cubic_interp1d(wave_flat, wave_grid, phase_interp[:, p]))(jnp.arange(len(phase_flat)))

    # Transpose and reshape to match input shapes
    result = result.T

    # Return scalar if both inputs are scalar
    if len(phase_shape) == 0 and len(wave_shape) == 0:
        return float(result)
    
    # Otherwise reshape to broadcast shape
    broadcast_shape = jnp.broadcast_shapes(phase_shape, wave_shape)
    return result.reshape(broadcast_shape)

@jax.jit
def salt2_colorlaw(wave, colorlaw_coeffs):
    """Calculate the SALT2 color law for the given wavelengths.

    Parameters
    ----------
    wave : array_like
        Wavelength values in Angstroms.
    colorlaw_coeffs : array_like
        Color law coefficients.

    Returns
    -------
    array_like
        Color law values.
    """
    B_WAVELENGTH = 4302.57  # B-band reference wavelength
    V_WAVELENGTH = 5428.55  # V-band reference wavelength
    colorlaw_range = (2800., 7000.)  # SALT2 color law range

    # Convert wavelengths to normalized wavelength
    v_minus_b = V_WAVELENGTH - B_WAVELENGTH
    l = (wave - B_WAVELENGTH) / v_minus_b
    l_lo = (colorlaw_range[0] - B_WAVELENGTH) / v_minus_b
    l_hi = (colorlaw_range[1] - B_WAVELENGTH) / v_minus_b

    # Calculate polynomial coefficients
    alpha = 1. - jnp.sum(colorlaw_coeffs)
    coeffs = jnp.concatenate([jnp.array([0., alpha]), colorlaw_coeffs])
    prime_coeffs = jnp.arange(len(coeffs)) * coeffs
    prime_coeffs = prime_coeffs[1:]  # Remove first element (0)

    # Calculate polynomial values at boundaries
    p_lo = jnp.polyval(jnp.flipud(coeffs), l_lo)
    pprime_lo = jnp.polyval(jnp.flipud(prime_coeffs), l_lo)
    p_hi = jnp.polyval(jnp.flipud(coeffs), l_hi)
    pprime_hi = jnp.polyval(jnp.flipud(prime_coeffs), l_hi)

    # Calculate extinction for each region
    extinction = jnp.where(
        l < l_lo,
        p_lo + pprime_lo * (l - l_lo),  # Blue side
        jnp.where(
            l > l_hi,
            p_hi + pprime_hi * (l - l_hi),  # Red side
            jnp.polyval(jnp.flipud(coeffs), l)  # In between
        )
    )

    return -extinction

@jax.jit
def salt2_flux(phase, wave, params):
    """Calculate SALT2 model flux at the given time and wavelength.

    Parameters
    ----------
    phase : array_like
        Rest-frame phase(s) in days.
    wave : array_like
        Rest-frame wavelength(s) in Angstroms.
    params : dict
        Model parameters including 'x0', 'x1', 'c', and 'z'.

    Returns
    -------
    array_like
        Rest-frame flux values in ergs/s/cm^2/Angstrom.
        Note: The redshift scaling (a^2) is applied by the Model class.
    """
    # Get parameters
    x0 = params['x0']
    x1 = params['x1']
    c = params['c']

    # Calculate rest-frame components
    m0 = salt2_m0(phase, wave)
    m1 = salt2_m1(phase, wave)

    # Calculate color law with actual SALT2 coefficients
    colorlaw_coeffs = jnp.array([-0.402687, 0.700296, -0.431342, 0.0779681])
    colorlaw = salt2_colorlaw(wave, colorlaw_coeffs)

    # Calculate rest-frame flux without redshift scaling
    # The redshift scaling (a^2) will be applied by the Model class
    return x0 * (m0 + x1 * m1) * 10**(-0.4 * colorlaw * c)