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
def kernval(x):
    """Bicubic convolution kernel used in SNCosmo.
    
    The kernel is defined by:
    W(x) = (a+2)*x^3-(a+3)*x^2+1 for x<=1
    W(x) = a(x^3-5*x^2+8*x-4) for 1<x<2
    W(x) = 0 for x>2
    where a = -0.5
    """
    A = -0.5
    B = A + 2.0  # 1.5
    C = A + 3.0  # 2.5
    x = jnp.abs(x)
    
    # Use jnp.where for vectorized conditional operations
    return jnp.where(
        x > 2.0,
        0.0,
        jnp.where(
            x < 1.0,
            x * x * (B * x - C) + 1.0,  # x^3 * (1.5) - x^2 * (2.5) + 1
            A * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0)  # -0.5(x^3 - 5x^2 + 8x - 4)
        )
    )

@jax.jit
def find_index(values, x):
    """Find index i such that values[i] <= x < values[i+1]."""
    i = jnp.searchsorted(values, x)
    i = jnp.clip(i, 1, len(values)-2)  # Need 2 points on each side
    return i.astype(jnp.int32)

@jax.jit
def bicubic_interp2d(x, y, xp, yp, zp):
    """2D bicubic convolution interpolation.
    
    Parameters
    ----------
    x : float or array_like
        x-coordinates at which to evaluate
    y : float or array_like
        y-coordinates at which to evaluate
    xp : array_like
        x-coordinates of the data points
    yp : array_like
        y-coordinates of the data points
    zp : array_like
        2D array of z values
    
    Returns
    -------
    array_like
        Interpolated values
    """
    # Convert inputs to arrays
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    
    # Find indices
    ix = find_index(xp, x)
    iy = find_index(yp, y)
    
    # Calculate normalized distances
    dx = (x - xp[ix]) / (xp[ix+1] - xp[ix])
    dy = (y - yp[iy]) / (yp[iy+1] - yp[iy])
    
    # Calculate weights
    wx = jnp.array([kernval(dx-1.0), kernval(dx), kernval(dx+1.0), kernval(dx+2.0)])
    wy = jnp.array([kernval(dy-1.0), kernval(dy), kernval(dy+1.0), kernval(dy+2.0)])
    
    # Handle boundary conditions
    x_near_boundary = (ix <= 1) | (ix >= len(xp)-3)
    y_near_boundary = (iy <= 1) | (iy >= len(yp)-3)
    
    # Linear interpolation result
    ax = (x - xp[ix]) / (xp[ix+1] - xp[ix])
    ay = (y - yp[iy]) / (yp[iy+1] - yp[iy])
    ay2 = 1.0 - ay
    
    linear_result = ((1.0 - ax) * (ay2 * zp[ix, iy] + ay * zp[ix, iy+1]) +
                     ax * (ay2 * zp[ix+1, iy] + ay * zp[ix+1, iy+1]))
    
    # Full bicubic convolution result
    cubic_result = 0.0
    for i in range(4):
        for j in range(4):
            cubic_result = cubic_result + wx[i] * wy[j] * zp[ix+i-1, iy+j-1]
    
    # Use linear interpolation near boundaries, cubic otherwise
    return jnp.where(x_near_boundary | y_near_boundary,
                    linear_result,
                    cubic_result)

@jax.jit
def salt2_m0(phase, wave):
    """Interpolate the M0 template at the given phase and wavelength.

    Args:
        phase (float or array-like): The phase(s) at which to evaluate the template.
        wave (float or array-like): The wavelength(s) at which to evaluate the template.

    Returns:
        array-like: The interpolated M0 template values.
    """
    # Convert inputs to arrays
    phase = jnp.asarray(phase)
    wave = jnp.asarray(wave)
    
    # Get original shapes
    phase_shape = phase.shape
    wave_shape = wave.shape
    
    # Reshape inputs to 1D
    phase_flat = jnp.ravel(phase)
    wave_flat = jnp.ravel(wave)
    
    # Create a mesh grid of phase and wave values
    phase_mesh, wave_mesh = jnp.meshgrid(phase_flat, wave_flat, indexing='ij')
    
    # Vectorize the interpolation over all phase/wave combinations
    result = jax.vmap(lambda p, w: bicubic_interp2d(p, w, phase_grid, wave_grid, M0_data))(
        jnp.ravel(phase_mesh), jnp.ravel(wave_mesh))
    
    # Reshape result to match input shapes
    if len(phase_shape) == 0 and len(wave_shape) == 0:
        return result[0]  # Return scalar for scalar inputs
    elif len(phase_shape) == 0:
        return result.reshape(wave_shape)  # Return 1D array for scalar phase
    elif len(wave_shape) == 0:
        return result.reshape(phase_shape)  # Return 1D array for scalar wavelength
    else:
        return result.reshape(phase_shape + wave_shape)  # Return shaped array for array inputs

@jax.jit
def salt2_m1(phase, wave):
    """Interpolate the M1 template at the given phase and wavelength.

    Args:
        phase (float or array-like): The phase(s) at which to evaluate the template.
        wave (float or array-like): The wavelength(s) at which to evaluate the template.

    Returns:
        array-like: The interpolated M1 template values.
    """
    # Convert inputs to arrays
    phase = jnp.asarray(phase)
    wave = jnp.asarray(wave)
    
    # Get original shapes
    phase_shape = phase.shape
    wave_shape = wave.shape
    
    # Reshape inputs to 1D
    phase_flat = jnp.ravel(phase)
    wave_flat = jnp.ravel(wave)
    
    # Create a mesh grid of phase and wave values
    phase_mesh, wave_mesh = jnp.meshgrid(phase_flat, wave_flat, indexing='ij')
    
    # Vectorize the interpolation over all phase/wave combinations
    result = jax.vmap(lambda p, w: bicubic_interp2d(p, w, phase_grid, wave_grid, M1_data))(
        jnp.ravel(phase_mesh), jnp.ravel(wave_mesh))
    
    # Reshape result to match input shapes
    if len(phase_shape) == 0 and len(wave_shape) == 0:
        return result[0]  # Return scalar for scalar inputs
    elif len(phase_shape) == 0:
        return result.reshape(wave_shape)  # Return 1D array for scalar phase
    elif len(wave_shape) == 0:
        return result.reshape(phase_shape)  # Return 1D array for scalar wavelength
    else:
        return result.reshape(phase_shape + wave_shape)  # Return shaped array for array inputs

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

    # Define polynomial evaluation function using Horner's method
    def polyval(coeffs, x):
        """Evaluate polynomial using Horner's method."""
        result = coeffs[0]
        for c in coeffs[1:]:
            result = result * x + c
        return result

    # Calculate polynomial values at boundaries
    coeffs_rev = jnp.flipud(coeffs)
    prime_coeffs_rev = jnp.flipud(prime_coeffs)
    
    p_lo = polyval(coeffs_rev, l_lo)
    pprime_lo = polyval(prime_coeffs_rev, l_lo)
    p_hi = polyval(coeffs_rev, l_hi)
    pprime_hi = polyval(prime_coeffs_rev, l_hi)

    # Calculate extinction for each region
    extinction = jnp.where(
        l < l_lo,
        p_lo + pprime_lo * (l - l_lo),  # Blue side
        jnp.where(
            l > l_hi,
            p_hi + pprime_hi * (l - l_hi),  # Red side
            polyval(coeffs_rev, l)  # In between
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

    # Calculate rest-frame flux
    # Note: SNCosmo applies the color law first, then scales by x0
    flux = x0 * (m0 + x1 * m1) * 10**(-0.4 * colorlaw * c)

    return flux