"""SALT3-NIR model implementation in JAX."""
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import sncosmo
import os
import pytest
import math
from jax_supernovae.bandpasses import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from functools import partial
from jax import vmap
import importlib.resources

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Constants
H_ERG_S = 6.62607015e-27  # Planck constant in erg*s

# Get package directory
PACKAGE_DIR = os.path.dirname(__file__)

# Model directory - hardcoded path
MODEL_DIR = os.path.join(PACKAGE_DIR, 'data/models/salt3-nir/salt3nir-p22')

def read_griddata_file(filename):
    """Read 2-d grid data from a text file.

    Each line has values `x0 x1 y` (phase, wavelength, flux). Space separated.
    Returns phase, wavelength arrays and a 2D grid of values.
    """
    # Read data from file
    data = np.loadtxt(filename)
    
    # Get unique phase and wavelength values, ensuring they're sorted
    phase = np.sort(np.unique(data[:, 0]))
    wave = np.sort(np.unique(data[:, 1]))
    
    # Create empty grid
    values = np.zeros((len(phase), len(wave)))
    
    # Map each data point to its position in the grid
    for p, w, v in data:
        pi = np.searchsorted(phase, p)
        wi = np.searchsorted(wave, w)
        values[pi, wi] = v
    
    return phase, wave, values

# Read M0 and M1 data
m0_file = os.path.join(MODEL_DIR, 'salt3_template_0.dat')
m1_file = os.path.join(MODEL_DIR, 'salt3_template_1.dat')
cl_file = os.path.join(MODEL_DIR, 'salt3_color_correction.dat')

# Read data and apply scaling (match SNCosmo exactly)
SCALE_FACTOR = 1e-12
phase_grid, wave_grid, m0_data = read_griddata_file(m0_file)
_, _, m1_data = read_griddata_file(m1_file)

# Apply scale factor to data (match SNCosmo exactly)
m0_data = m0_data * SCALE_FACTOR
m1_data = m1_data * SCALE_FACTOR

# Convert to JAX arrays
phase_grid = jnp.array(phase_grid)
wave_grid = jnp.array(wave_grid)
m0_data = jnp.array(m0_data)  # Scale factor already applied
m1_data = jnp.array(m1_data)  # Scale factor already applied

# Read color law coefficients
with open(cl_file, 'r') as f:
    words = f.read().split()
    ncoeffs = int(words[0])
    colorlaw_coeffs = jnp.array([float(word) for word in words[1: 1 + ncoeffs]])
    colorlaw_range = [3000., 7000.]  # Default range
    for i in range(1+ncoeffs, len(words), 2):
        if words[i] == 'Salt2ExtinctionLaw.min_lambda':
            colorlaw_range[0] = float(words[i+1])
        elif words[i] == 'Salt2ExtinctionLaw.max_lambda':
            colorlaw_range[1] = float(words[i+1])

@jax.jit
def kernval(x):
    """Compute kernel value for bicubic interpolation.
    
    This matches SNCosmo's implementation exactly:
    W(x) = (a+2)*x**3-(a+3)*x**2+1 for x<=1
    W(x) = a( x**3-5*x**2+8*x-4) for 1<x<2
    W(x) = 0 for x>2
    where a=-0.5
    """
    x = jnp.abs(x)
    a = -0.5  # This matches SNCosmo's value
    
    # Calculate the result for each case
    case1 = (a + 2) * x**3 - (a + 3) * x**2 + 1  # x <= 1
    case2 = a * (x**3 - 5 * x**2 + 8 * x - 4)    # 1 < x < 2
    
    # Use where to select the appropriate result
    result = jnp.where(x <= 1, case1,
                      jnp.where(x < 2, case2, 0.0))
    
    return result

@jax.jit
def find_index(values, x):
    """Find index i such that values[i] <= x < values[i+1]."""
    i = jnp.searchsorted(values, x) - 1
    i = jnp.clip(i, 0, len(values) - 2)  # Ensure we stay within bounds
    return i.astype(jnp.int32)

@jax.jit
def compute_interpolation_weights(x, values):
    """Compute interpolation weights and indices.
    
    Args:
        x: The point to interpolate at
        values: The grid values to interpolate between
        
    Returns:
        tuple: (indices, normalized coordinates, near_boundary)
    """
    # Find indices
    i = find_index(values, x)
    
    # Check bounds and boundaries
    in_bounds = (x >= values[0]) & (x <= values[-1])
    near_boundary = (i <= 0) | (i >= len(values) - 2)
    
    # Calculate normalized coordinates
    dx = (x - values[i]) / (values[i + 1] - values[i])
    
    return i, dx, in_bounds, near_boundary

@jax.jit
def interpolate_2d(phase, wave, data):
    """Perform 2D interpolation on gridded data.
    
    Args:
        phase: Phase value to interpolate at
        wave: Wavelength value to interpolate at
        data: 2D grid of values to interpolate from
        
    Returns:
        float: Interpolated value
    """
    # Compute weights for both dimensions
    ix, dx, x_in_bounds, x_near_boundary = compute_interpolation_weights(phase, phase_grid)
    iy, dy, y_in_bounds, y_near_boundary = compute_interpolation_weights(wave, wave_grid)
    
    # Check if we need to use linear interpolation
    near_boundary = x_near_boundary | y_near_boundary
    
    # Get corner values for linear interpolation
    z00 = data[ix, iy]
    z01 = data[ix, iy + 1]
    z10 = data[ix + 1, iy]
    z11 = data[ix + 1, iy + 1]
    
    # Linear interpolation
    linear_result = (z00 * (1 - dx) * (1 - dy) +
                    z10 * dx * (1 - dy) +
                    z01 * (1 - dx) * dy +
                    z11 * dx * dy)
    
    # For bicubic interpolation, pad the array with edge values
    padded = jnp.pad(data, ((1, 1), (1, 1)), mode='edge')
    
    # Get 4x4 grid for bicubic interpolation
    ix_pad = ix + 1  # Adjust for padding
    iy_pad = iy + 1
    grid = lax.dynamic_slice(padded, (ix_pad - 1, iy_pad - 1), (4, 4))
    
    # Calculate bicubic weights
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
    
    # Calculate bicubic interpolation
    cubic_result = jnp.sum(jnp.outer(wx, wy) * grid)
    
    # Use linear interpolation near boundaries, bicubic otherwise
    result = jnp.where(near_boundary, linear_result, cubic_result)
    
    # Return 0 if out of bounds, interpolated value otherwise
    return jnp.where(x_in_bounds & y_in_bounds, result, 0.0)

@jax.jit
def salt3_m0_single(phase, wave):
    """Get the M0 component at a single phase and wavelength."""
    return interpolate_2d(phase, wave, m0_data)

@jax.jit
def salt3_m1_single(phase, wave):
    """Get the M1 component at a single phase and wavelength."""
    return interpolate_2d(phase, wave, m1_data)

@jax.jit
def salt3_m0(phase, wave):
    """Get the M0 component at the given phase and wavelength.

    Args:
        phase (float or array): Rest-frame phase in days
        wave (float or array): Rest-frame wavelength in Angstroms

    Returns:
        float or array: M0 component value(s)
    """
    phase = jnp.asarray(phase)
    wave = jnp.asarray(wave)

    # Handle scalar inputs
    if phase.ndim == 0 and wave.ndim == 0:
        return salt3_m0_single(phase, wave)

    # Handle 2D inputs with broadcasting
    if phase.ndim == 2 and wave.ndim == 2:
        # First vmap over phases (axis 0)
        phase_mapped = jax.vmap(lambda p: jax.vmap(lambda w: salt3_m0_single(p, w))(wave[0, :]))(phase[:, 0])
        return phase_mapped

    # Handle array inputs of same size
    if phase.ndim == 1 and wave.ndim == 1 and phase.shape == wave.shape:
        return jax.vmap(lambda p, w: salt3_m0_single(p, w))(phase, wave)

    # Handle broadcasting case (phase array with single wavelength)
    if phase.ndim == 1 and wave.ndim == 0:
        return jax.vmap(lambda p: salt3_m0_single(p, wave))(phase)

    # Handle broadcasting case (single phase with wavelength array)
    if phase.ndim == 0 and wave.ndim == 1:
        return jax.vmap(lambda w: salt3_m0_single(phase, w))(wave)

    # Handle broadcasting case (phase array with wave array of different size)
    if phase.ndim == 1 and wave.ndim == 1:
        # First map over phases, then over wavelengths
        return jax.vmap(lambda p: jax.vmap(lambda w: salt3_m0_single(p, w))(wave))(phase)

    raise ValueError("Unsupported input shapes for salt3_m0")

@jax.jit
def salt3_m1(phase, wave):
    """Get the M1 component at the given phase and wavelength.

    Args:
        phase (float or array): Rest-frame phase in days
        wave (float or array): Rest-frame wavelength in Angstroms

    Returns:
        float or array: M1 component value(s)
    """
    phase = jnp.asarray(phase)
    wave = jnp.asarray(wave)

    # Handle scalar inputs
    if phase.ndim == 0 and wave.ndim == 0:
        return salt3_m1_single(phase, wave)

    # Handle 2D inputs with broadcasting
    if phase.ndim == 2 and wave.ndim == 2:
        # First vmap over phases (axis 0)
        phase_mapped = jax.vmap(lambda p: jax.vmap(lambda w: salt3_m1_single(p, w))(wave[0, :]))(phase[:, 0])
        return phase_mapped

    # Handle array inputs of same size
    if phase.ndim == 1 and wave.ndim == 1 and phase.shape == wave.shape:
        return jax.vmap(lambda p, w: salt3_m1_single(p, w))(phase, wave)

    # Handle broadcasting case (phase array with single wavelength)
    if phase.ndim == 1 and wave.ndim == 0:
        return jax.vmap(lambda p: salt3_m1_single(p, wave))(phase)

    # Handle broadcasting case (single phase with wavelength array)
    if phase.ndim == 0 and wave.ndim == 1:
        return jax.vmap(lambda w: salt3_m1_single(phase, w))(wave)

    # Handle broadcasting case (phase array with wave array of different size)
    if phase.ndim == 1 and wave.ndim == 1:
        # First map over phases, then over wavelengths
        return jax.vmap(lambda p: jax.vmap(lambda w: salt3_m1_single(p, w))(wave))(phase)

    raise ValueError("Unsupported input shapes for salt3_m1")

@jax.jit
def salt3_colorlaw(wave):
    """Calculate SALT3 color law at given wavelength."""
    wave = jnp.asarray(wave)
    
    # Define constants (exactly as in SNCosmo)
    B_WAVE = 4302.57
    V_WAVE = 5428.55
    v_minus_b = V_WAVE - B_WAVE
    
    # Calculate normalized wavelength
    l = (wave - B_WAVE) / v_minus_b
    l_lo = (colorlaw_range[0] - B_WAVE) / v_minus_b
    l_hi = (colorlaw_range[1] - B_WAVE) / v_minus_b
    
    # Calculate polynomial coefficients
    alpha = 1. - jnp.sum(colorlaw_coeffs)
    coeffs = jnp.concatenate([jnp.array([0., alpha]), colorlaw_coeffs])
    coeffs_rev = jnp.flipud(coeffs)
    
    # Calculate derivative coefficients
    prime_coeffs = jnp.arange(len(coeffs)) * coeffs
    prime_coeffs = prime_coeffs[1:]  # Remove first element (0)
    prime_coeffs_rev = jnp.flipud(prime_coeffs)
    
    # Calculate polynomial values at boundaries
    p_lo = jnp.polyval(coeffs_rev, l_lo)
    pprime_lo = jnp.polyval(prime_coeffs_rev, l_lo)
    p_hi = jnp.polyval(coeffs_rev, l_hi)
    pprime_hi = jnp.polyval(prime_coeffs_rev, l_hi)
    
    # Calculate extinction for each region
    extinction = jnp.where(
        l < l_lo,
        p_lo + pprime_lo * (l - l_lo),  # Blue side
        jnp.where(
            l > l_hi,
            p_hi + pprime_hi * (l - l_hi),  # Red side
            jnp.polyval(coeffs_rev, l)  # In between
        )
    )
    
    # Return negative extinction to match SNCosmo's convention
    return -extinction

@partial(jax.jit, static_argnames=['bandpass', 'zpsys'])
def salt3_bandflux(phase, bandpass, params, zp=None, zpsys=None):
    """Calculate bandflux for SALT3 model.
    
    Parameters
    ----------
        Rest-frame phase in days relative to maximum brightness.
    bandpass : Bandpass object
        Bandpass to calculate flux through.
    params : dict
        Model parameters including z, t0, x0, x1, c.
    zp : float or None, optional
        Zero point for flux. If None, no scaling is applied.
    zpsys : str, optional
        Magnitude system for zero point. Must be provided if zp is not None.
        Default is None.
        
    Returns
    -------
    float or array_like
        Flux in photons/s/cm^2. Return value is float if phase is scalar,
        array if phase is array. If zp and zpsys are given, flux is scaled
        to the requested zeropoint.
    """
    # Check that if zp is provided, zpsys must also be provided
    if zp is not None and zpsys is None:
        raise ValueError('zpsys must be given if zp is not None')
    
    # Convert inputs to arrays and check if input was scalar
    phase = jnp.atleast_1d(phase)
    is_scalar = len(phase.shape) == 0
    
    # Get parameters
    z = params['z']
    t0 = params['t0']
    x0 = params['x0']
    x1 = params['x1']
    c = params['c']
    
    # Convert to rest-frame phase
    a = 1.0 / (1.0 + z)  # Scale factor
    restphase = (phase - t0) * a
    
    # Use pre-computed integration grid from bandpass
    wave = bandpass.integration_wave
    dwave = bandpass.integration_spacing
    restwave = wave * a
    trans = bandpass(wave)
    
    # Pre-compute color law for all wavelengths
    cl = salt3_colorlaw(restwave)
    
    # Compute M0 and M1 components for all phases and wavelengths at once
    m0 = salt3_m0(restphase[:, None], restwave[None, :])
    m1 = salt3_m1(restphase[:, None], restwave[None, :])
    
    # Calculate rest-frame flux for all phases and wavelengths at once
    rest_flux = x0 * (m0 + x1 * m1) * 10**(-0.4 * cl[None, :] * c) * a
    
    # Integrate flux through bandpass using trapezoidal rule
    # Multiply by wave and transmission before integration
    integrand = wave[None, :] * trans[None, :] * rest_flux
    result = jnp.trapezoid(integrand, wave, axis=1) / HC_ERG_AA
    
    # Apply zero point if provided
    if zp is not None:
        # Get the magsystem's zpbandflux for this bandpass
        if zpsys == 'ab':
            # For AB system, calculate zpbandflux
            # AB spectrum is 3631 x 10^{-23} erg/s/cm^2/Hz
            # Convert to F_lambda: 3631e-23 * c / wave^2 erg/s/cm^2/AA
            # Then integrate: sum(f * trans * wave) * dwave / (hc)
            zpbandflux = 3631e-23 * dwave / H_ERG_S * jnp.sum(trans / wave)
        else:
            raise ValueError(f"Unsupported magnitude system: {zpsys}")
        
        # Scale the flux according to the zeropoint (exactly like sncosmo)
        zpnorm = 10.**(0.4 * zp) / zpbandflux
        result = result * zpnorm
    
    # Return scalar if input was scalar
    if is_scalar:
        result = result[0]
    
    return result

@partial(jax.jit, static_argnames=['bandpasses', 'zpsys'])
def salt3_multiband_flux(phase, bandpasses, params, zps=None, zpsys=None):
    """Calculate flux for multiple bandpasses at once.
    
    Args:
        phase (array-like): Phase(s) in observer frame.
        bandpasses (list): List of Bandpass objects.
        params (dict): Model parameters including z, t0, x0, x1, c.
        zps (array-like, optional): Zero points for each bandpass.
        zpsys (str, optional): Magnitude system (e.g. 'ab').
        
    Returns:
        array-like: Flux values for each phase and bandpass combination.
    """
    # Convert inputs to arrays
    phase = jnp.atleast_1d(phase)
    n_phase = len(phase)
    n_bands = len(bandpasses)
    
    # Initialize output array
    result = jnp.zeros((n_phase, n_bands))
    
    # Calculate flux for each bandpass
    for i in range(n_bands):
        zp = zps[i] if zps is not None else None
        band_flux = salt3_bandflux(phase, bandpasses[i], params, zp=zp, zpsys=zpsys)
        result = result.at[:, i].set(band_flux)
    
    return result 

def precompute_bandflux_bridge(bandpass):
    """
    Precompute static components for a given bandpass.
    
    Returns a dictionary containing:
        - 'wave': the integration wavelength grid,
        - 'dwave': spacing between grid points,
        - 'trans': the transmission values computed on the grid.
    """
    wave = bandpass.integration_wave
    dwave = bandpass.integration_spacing
    trans = bandpass(wave)
    return {'wave': wave, 'dwave': dwave, 'trans': trans}

@partial(jax.jit, static_argnames=['zpsys'])
def optimized_salt3_bandflux(phase, wave, dwave, trans, params, zp=None, zpsys=None):
    """
    Calculate bandflux for a single bandpass using precomputed static data.
    
    Parameters:
        phase : array or scalar
            Observer-frame phase(s) at which to compute the flux.
        wave : array
            Wavelength grid for integration.
        dwave : float
            Spacing between wavelength grid points.
        trans : array
            Transmission values on the wavelength grid.
        params : dict
            Dictionary containing model parameters: 'z', 't0', 'x0', 'x1', 'c'.
        zp : float or None
            Zero point for flux scaling. If provided, zpsys must also be given.
        zpsys : str or None
            Magnitude system (e.g. 'ab').
    
    Returns:
        Flux in photons/s/cm^2. If phase is scalar then returns scalar.
    """
    if zp is not None and zpsys is None:
        raise ValueError('zpsys must be given if zp is not None')

    # Convert inputs to arrays and check if input was scalar
    phase = jnp.atleast_1d(phase)
    is_scalar = len(phase.shape) == 0

    z = params['z']
    t0 = params['t0']
    x0 = params['x0']
    x1 = params['x1']
    c  = params['c']

    # Calculate scaling factor and transform phase to rest-frame.
    a = 1.0 / (1.0 + z)
    restphase = (phase - t0) * a

    # Scale the integration grid to rest-frame wavelengths.
    restwave = wave * a
    # Compute colour law on the restwave grid.
    cl = salt3_colorlaw(restwave)

    # Compute m0 and m1 components over the 2D grid.
    m0 = salt3_m0(restphase[:, None], restwave[None, :])
    m1 = salt3_m1(restphase[:, None], restwave[None, :])

    # Compute rest-frame flux including the colour law effect.
    rest_flux = x0 * (m0 + x1 * m1) * 10**(-0.4 * cl[None, :] * c) * a
    integrand = wave[None, :] * trans[None, :] * rest_flux
    result = jnp.trapezoid(integrand, wave, axis=1) / HC_ERG_AA

    # Apply zero point correction if required.
    if zp is not None:
        if zpsys == 'ab':
            zpbandflux = 3631e-23 * dwave / H_ERG_S * jnp.sum(trans / wave)
        else:
            raise ValueError(f"Unsupported magnitude system: {zpsys}")
        zpnorm = 10**(0.4 * zp) / zpbandflux
        result = result * zpnorm

    # Return scalar if input was scalar
    if is_scalar:
        result = result[0]
    return result

@partial(jax.jit, static_argnames=['zpsys'])
def optimized_salt3_multiband_flux(phase, bridges, params, zps=None, zpsys=None):
    """
    Calculate fluxes for multiple bandpasses.
    
    Parameters:
        phase : array
            Observer-frame phases.
        bridges : list of dict
            Precomputed bridge data for each bandpass.
        params : dict
            Model parameters.
        zps : list or array or None
            Zero points for each bandpass.
        zpsys : str or None
            Magnitude system.
    
    Returns:
        Array of flux values for each phase and band.
    """
    phase = jnp.atleast_1d(phase)
    n_phase = len(phase)
    n_bands = len(bridges)
    result = jnp.zeros((n_phase, n_bands))
    
    for i in range(n_bands):
        bp_bridge = bridges[i]
        curr_zp = zps[i] if zps is not None else None
        band_flux = optimized_salt3_bandflux(
            phase, 
            bp_bridge['wave'], 
            bp_bridge['dwave'], 
            bp_bridge['trans'], 
            params, 
            zp=curr_zp, 
            zpsys=zpsys
        )
        result = result.at[:, i].set(band_flux)
    
    return result 