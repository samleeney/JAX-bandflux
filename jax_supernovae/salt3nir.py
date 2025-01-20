"""SALT3-NIR model implementation in JAX."""
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import sncosmo
import os
import pytest
import math
from jax_supernovae.core import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from functools import partial
from jax import vmap

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

def integration_grid(low, high, target_spacing):
    """Create a wavelength grid for integration.
    
    Args:
        low: Minimum wavelength
        high: Maximum wavelength
        target_spacing: Target spacing between wavelength points
        
    Returns:
        wave: Array of wavelength points
        spacing: Actual spacing between points
    """
    # Convert inputs to concrete values
    low = float(low)
    high = float(high)
    target_spacing = float(target_spacing)
    
    # Calculate range difference and spacing (match SNCosmo exactly)
    range_diff = high - low
    spacing = range_diff / int(np.ceil(range_diff / target_spacing))
    
    # Create grid using numpy then convert to JAX array (match SNCosmo exactly)
    wave = np.arange(low + 0.5 * spacing, high, spacing)
    wave = jnp.array(wave)
    
    return wave, spacing

# Get model files from project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'sncosmo-modelfiles/models/salt3-nir/salt3nir-p22')

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
def bicubic_interp2d(x, y, xp, yp, zp):
    """2D bicubic convolution interpolation."""
    # Convert inputs to arrays and ensure they are scalars
    x = jnp.asarray(x).reshape(-1)[0]  # Take first element if array
    y = jnp.asarray(y).reshape(-1)[0]  # Take first element if array
    
    # Find indices
    ix = find_index(xp, x)
    iy = find_index(yp, y)
    
    # Convert indices to scalars
    ix = jnp.asarray(ix).reshape(-1)[0]
    iy = jnp.asarray(iy).reshape(-1)[0]
    
    # Check bounds
    x_in_bounds = (x >= xp[0]) & (x <= xp[-1])
    y_in_bounds = (y >= yp[0]) & (y <= yp[-1])
    
    # Check if we're near boundaries (exactly like SNCosmo)
    x_near_boundary = (ix <= 0) | (ix >= len(xp) - 2)
    y_near_boundary = (iy <= 0) | (iy >= len(yp) - 2)
    near_boundary = x_near_boundary | y_near_boundary
    
    # Calculate normalized coordinates (exactly like SNCosmo)
    dx = (x - xp[ix]) / (xp[ix + 1] - xp[ix])
    dy = (y - yp[iy]) / (yp[iy + 1] - yp[iy])
    
    # Get corner values for linear interpolation using dynamic_slice
    z00 = zp[ix, iy]
    z01 = zp[ix, iy + 1]
    z10 = zp[ix + 1, iy]
    z11 = zp[ix + 1, iy + 1]
    
    # Linear interpolation
    linear_result = (z00 * (1 - dx) * (1 - dy) +
                    z10 * dx * (1 - dy) +
                    z01 * (1 - dx) * dy +
                    z11 * dx * dy)
    
    # For bicubic interpolation, pad the array with edge values
    padded = jnp.pad(zp, ((1, 1), (1, 1)), mode='edge')
    
    # Get 4x4 grid for bicubic interpolation
    ix_pad = ix + 1  # Adjust for padding
    iy_pad = iy + 1
    
    # Use dynamic_slice to get the grid
    grid = lax.dynamic_slice(padded, (ix_pad - 1, iy_pad - 1), (4, 4))
    
    # Calculate bicubic weights
    wx = jnp.array([
        kernval(dx + 1.0),  # For ix-1
        kernval(dx),        # For ix
        kernval(dx - 1.0),  # For ix+1
        kernval(dx - 2.0)   # For ix+2
    ])
    
    wy = jnp.array([
        kernval(dy + 1.0),  # For iy-1
        kernval(dy),        # For iy
        kernval(dy - 1.0),  # For iy+1
        kernval(dy - 2.0)   # For iy+2
    ])
    
    # Calculate bicubic interpolation
    cubic_result = jnp.sum(jnp.outer(wx, wy) * grid)
    
    # Use linear interpolation near boundaries, bicubic otherwise
    result = jnp.where(near_boundary, linear_result, cubic_result)
    
    # Return 0 if out of bounds, interpolated value otherwise
    return jnp.where(x_in_bounds & y_in_bounds, result, 0.0)

@partial(vmap, in_axes=(0, None, None))
def interp(x, xp, fp):
    """Simple linear interpolation based on jax-cosmo implementation.
    
    Args:
        x: Points at which to evaluate the interpolation
        xp: x-coordinates of data points (must be sorted)
        fp: y-coordinates of data points
        
    Returns:
        Interpolated values at x
    """
    # Find nearest neighbor index
    ind = jnp.searchsorted(xp, x) - 1
    
    # Clip to valid range
    ind = jnp.clip(ind, 0, len(xp) - 2)
    
    # Get x values and slopes
    x0 = xp[ind]
    x1 = xp[ind + 1]
    y0 = fp[ind]
    y1 = fp[ind + 1]
    
    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)

@jax.jit
def get_grid_values(data, ix, iy):
    """Get a 4x4 grid of values around the given indices."""
    # Get the starting indices for the 4x4 grid
    ix_start = ix - 2
    iy_start = iy - 2
    
    # Use dynamic_slice to get the values
    grid = jax.lax.dynamic_slice(
        data,
        (ix_start, iy_start),
        (4, 4)
    )
    
    # Return grid values (already scaled)
    return grid

@jax.jit
def salt3nir_m0_single(phase, wave):
    """Get the M0 component at a single phase and wavelength.
    
    Args:
        phase (float): Rest-frame phase in days
        wave (float): Rest-frame wavelength in Angstroms
        
    Returns:
        float: M0 component value (scale factor already applied when loading data)
    """
    # Find indices
    ix = find_index(phase_grid, phase)
    iy = find_index(wave_grid, wave)
    
    # Check bounds
    x_in_bounds = (phase >= phase_grid[0]) & (phase <= phase_grid[-1])
    y_in_bounds = (wave >= wave_grid[0]) & (wave <= wave_grid[-1])
    
    # Check if we're near boundaries (exactly like SNCosmo)
    x_near_boundary = (ix <= 0) | (ix >= len(phase_grid) - 2)
    y_near_boundary = (iy <= 0) | (iy >= len(wave_grid) - 2)
    near_boundary = x_near_boundary | y_near_boundary
    
    # Calculate normalized coordinates (exactly like SNCosmo)
    dx = (phase - phase_grid[ix]) / (phase_grid[ix + 1] - phase_grid[ix])
    dy = (wave - wave_grid[iy]) / (wave_grid[iy + 1] - wave_grid[iy])
    
    # Get corner values for linear interpolation using dynamic_slice
    z00 = m0_data[ix, iy]
    z01 = m0_data[ix, iy + 1]
    z10 = m0_data[ix + 1, iy]
    z11 = m0_data[ix + 1, iy + 1]
    
    # Linear interpolation
    linear_result = (z00 * (1 - dx) * (1 - dy) +
                    z10 * dx * (1 - dy) +
                    z01 * (1 - dx) * dy +
                    z11 * dx * dy)
    
    # For bicubic interpolation, pad the array with edge values
    padded = jnp.pad(m0_data, ((1, 1), (1, 1)), mode='edge')
    
    # Get 4x4 grid for bicubic interpolation
    ix_pad = ix + 1  # Adjust for padding
    iy_pad = iy + 1
    
    # Use dynamic_slice to get the grid
    grid = lax.dynamic_slice(padded, (ix_pad - 1, iy_pad - 1), (4, 4))
    
    # Calculate bicubic weights
    wx = jnp.array([
        kernval(dx + 1.0),  # For ix-1
        kernval(dx),        # For ix
        kernval(dx - 1.0),  # For ix+1
        kernval(dx - 2.0)   # For ix+2
    ])
    
    wy = jnp.array([
        kernval(dy + 1.0),  # For iy-1
        kernval(dy),        # For iy
        kernval(dy - 1.0),  # For iy+1
        kernval(dy - 2.0)   # For iy+2
    ])
    
    # Calculate bicubic interpolation
    cubic_result = jnp.sum(jnp.outer(wx, wy) * grid)
    
    # Use linear interpolation near boundaries, bicubic otherwise
    result = jnp.where(near_boundary, linear_result, cubic_result)
    
    # Return 0 if out of bounds, interpolated value otherwise
    return jnp.where(x_in_bounds & y_in_bounds, result, 0.0)

@jax.jit
def salt3nir_m0(phase, wave):
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
        return salt3nir_m0_single(phase, wave)

    # Handle 2D inputs with broadcasting
    if phase.ndim == 2 and wave.ndim == 2:
        # First vmap over phases (axis 0)
        phase_mapped = jax.vmap(lambda p: jax.vmap(lambda w: salt3nir_m0_single(p, w))(wave[0, :]))(phase[:, 0])
        return phase_mapped

    # Handle array inputs of same size
    if phase.ndim == 1 and wave.ndim == 1 and phase.shape == wave.shape:
        return jax.vmap(lambda p, w: salt3nir_m0_single(p, w))(phase, wave)

    # Handle broadcasting case (phase array with single wavelength)
    if phase.ndim == 1 and wave.ndim == 0:
        return jax.vmap(lambda p: salt3nir_m0_single(p, wave))(phase)

    # Handle broadcasting case (single phase with wavelength array)
    if phase.ndim == 0 and wave.ndim == 1:
        return jax.vmap(lambda w: salt3nir_m0_single(phase, w))(wave)

    # Handle broadcasting case (phase array with wave array of different size)
    if phase.ndim == 1 and wave.ndim == 1:
        # First map over phases, then over wavelengths
        return jax.vmap(lambda p: jax.vmap(lambda w: salt3nir_m0_single(p, w))(wave))(phase)

    raise ValueError("Unsupported input shapes for salt3nir_m0")

@jax.jit
def salt3nir_m1_single(phase, wave):
    """Get the M1 component at a single phase and wavelength.
    
    Args:
        phase (float): Rest-frame phase in days
        wave (float): Rest-frame wavelength in Angstroms
        
    Returns:
        float: M1 component value (scale factor already applied when loading data)
    """
    # Find indices
    ix = find_index(phase_grid, phase)
    iy = find_index(wave_grid, wave)
    
    # Check bounds
    x_in_bounds = (phase >= phase_grid[0]) & (phase <= phase_grid[-1])
    y_in_bounds = (wave >= wave_grid[0]) & (wave <= wave_grid[-1])
    
    # Check if we're near boundaries (exactly like SNCosmo)
    x_near_boundary = (ix <= 0) | (ix >= len(phase_grid) - 2)
    y_near_boundary = (iy <= 0) | (iy >= len(wave_grid) - 2)
    near_boundary = x_near_boundary | y_near_boundary
    
    # Calculate normalized coordinates (exactly like SNCosmo)
    dx = (phase - phase_grid[ix]) / (phase_grid[ix + 1] - phase_grid[ix])
    dy = (wave - wave_grid[iy]) / (wave_grid[iy + 1] - wave_grid[iy])
    
    # Get corner values for linear interpolation using dynamic_slice
    z00 = m1_data[ix, iy]
    z01 = m1_data[ix, iy + 1]
    z10 = m1_data[ix + 1, iy]
    z11 = m1_data[ix + 1, iy + 1]
    
    # Linear interpolation
    linear_result = (z00 * (1 - dx) * (1 - dy) +
                    z10 * dx * (1 - dy) +
                    z01 * (1 - dx) * dy +
                    z11 * dx * dy)
    
    # For bicubic interpolation, pad the array with edge values
    padded = jnp.pad(m1_data, ((1, 1), (1, 1)), mode='edge')
    
    # Get 4x4 grid for bicubic interpolation
    ix_pad = ix + 1  # Adjust for padding
    iy_pad = iy + 1
    
    # Use dynamic_slice to get the grid
    grid = lax.dynamic_slice(padded, (ix_pad - 1, iy_pad - 1), (4, 4))
    
    # Calculate bicubic weights
    wx = jnp.array([
        kernval(dx + 1.0),  # For ix-1
        kernval(dx),        # For ix
        kernval(dx - 1.0),  # For ix+1
        kernval(dx - 2.0)   # For ix+2
    ])
    
    wy = jnp.array([
        kernval(dy + 1.0),  # For iy-1
        kernval(dy),        # For iy
        kernval(dy - 1.0),  # For iy+1
        kernval(dy - 2.0)   # For iy+2
    ])
    
    # Calculate bicubic interpolation
    cubic_result = jnp.sum(jnp.outer(wx, wy) * grid)
    
    # Use linear interpolation near boundaries, bicubic otherwise
    result = jnp.where(near_boundary, linear_result, cubic_result)
    
    # Return 0 if out of bounds, interpolated value otherwise
    return jnp.where(x_in_bounds & y_in_bounds, result, 0.0)

@jax.jit
def salt3nir_m1(phase, wave):
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
        return salt3nir_m1_single(phase, wave)

    # Handle 2D inputs with broadcasting
    if phase.ndim == 2 and wave.ndim == 2:
        # First vmap over phases (axis 0)
        phase_mapped = jax.vmap(lambda p: jax.vmap(lambda w: salt3nir_m1_single(p, w))(wave[0, :]))(phase[:, 0])
        return phase_mapped

    # Handle array inputs of same size
    if phase.ndim == 1 and wave.ndim == 1 and phase.shape == wave.shape:
        return jax.vmap(lambda p, w: salt3nir_m1_single(p, w))(phase, wave)

    # Handle broadcasting case (phase array with single wavelength)
    if phase.ndim == 1 and wave.ndim == 0:
        return jax.vmap(lambda p: salt3nir_m1_single(p, wave))(phase)

    # Handle broadcasting case (single phase with wavelength array)
    if phase.ndim == 0 and wave.ndim == 1:
        return jax.vmap(lambda w: salt3nir_m1_single(phase, w))(wave)

    # Handle broadcasting case (phase array with wave array of different size)
    if phase.ndim == 1 and wave.ndim == 1:
        # First map over phases, then over wavelengths
        return jax.vmap(lambda p: jax.vmap(lambda w: salt3nir_m1_single(p, w))(wave))(phase)

    raise ValueError("Unsupported input shapes for salt3nir_m1")

@jax.jit
def salt3nir_colorlaw(wave):
    """Calculate SALT3-NIR color law at given wavelength."""
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

@jax.jit
def salt3nir_flux(phase, wave, params):
    """Calculate flux for SALT3-NIR model at given phase and wavelength."""
    # Get components
    m0 = salt3nir_m0(phase, wave)
    m1 = salt3nir_m1(phase, wave)
    cl = salt3nir_colorlaw(wave)
    
    # Debug: Print intermediate values
    print(f"Debug: phase={phase}, wave={wave}, m0={m0}, m1={m1}, cl={cl}")
    print(f"Debug: params={params}")
    
    # Calculate flux
    flux = params['x0'] * (m0 + params['x1'] * m1) * 10**(-0.4 * cl * params['c'])
    print(f"Debug: Calculated flux={flux}")
    return flux

# Define test phases and wavelengths
phase_test = jnp.array([0.0, 10.0, 20.0])
wave_test = jnp.array([4000.0, 5000.0, 6000.0])

# Debug: Review initial data values and interpolation process for M0
print("Debug: Reviewing initial data values and interpolation process for M0...")
# Check initial M0 data values
print(f"Initial M0 data (first 5 values): {m0_data.flatten()[:5]}")
# Check interpolation process at specific points
for p in phase_test:
    for w in wave_test:
        interpolated_value = salt3nir_m0(p, w)
        print(f"Interpolated M0 at phase {p}, wave {w}: {interpolated_value}")
print("Debug: Initial data values and interpolation process review complete.")

# Debug: Compare m0 values with SNCosmo
print("Debug: Comparing m0 values with SNCosmo...")
for p in phase_test:
    for w in wave_test:
        snc_m0 = sncosmo.Model(source='salt3-nir')._source._model['M0'](np.array([p]), np.array([w]))[0][0]
        jax_m0 = salt3nir_m0(p, w)
        print(f"Phase {p}, Wave {w} - SNCosmo M0: {snc_m0}, JAX M0: {jax_m0}")
print("Debug: m0 comparison with SNCosmo complete.")

# Debug: Compare m1 values with SNCosmo
print("Debug: Comparing m1 values with SNCosmo...")
for p in phase_test:
    for w in wave_test:
        snc_m1 = sncosmo.Model(source='salt3-nir')._source._model['M1'](np.array([p]), np.array([w]))[0][0]
        jax_m1 = salt3nir_m1(p, w)
        print(f"Phase {p}, Wave {w} - SNCosmo M1: {snc_m1}, JAX M1: {jax_m1}")
print("Debug: m1 comparison with SNCosmo complete.")

# Debug: Compare color law values with SNCosmo
print("Debug: Comparing color law values with SNCosmo...")
for w in wave_test:
    snc_cl = sncosmo.Model(source='salt3-nir').source.colorlaw(np.array([w]))
    jax_cl = salt3nir_colorlaw(w)
    print(f"Wave {w} - SNCosmo CL: {snc_cl}, JAX CL: {jax_cl}")
print("Debug: color law comparison with SNCosmo complete.")

# Test function for M0 component
@pytest.mark.parametrize("phase, wave", [
    (0.0, 4000.0),
    (10.0, 5000.0),
    (20.0, 6000.0)
])
def test_salt3nir_m0(phase, wave):
    snc_m0 = sncosmo.Model(source='salt3-nir')._source._model['M0'](np.array([phase]), np.array([wave]))[0][0]
    jax_m0 = salt3nir_m0(phase, wave)
    assert jnp.isclose(jax_m0, snc_m0, atol=1e-5), f"Mismatch for M0 at phase {phase}, wave {wave}: SNCosmo {snc_m0}, JAX {jax_m0}"

# Test function for M1 component
@pytest.mark.parametrize("phase, wave", [
    (0.0, 4000.0),
    (10.0, 5000.0),
    (20.0, 6000.0)
])
def test_salt3nir_m1(phase, wave):
    snc_m1 = sncosmo.Model(source='salt3-nir')._source._model['M1'](np.array([phase]), np.array([wave]))[0][0]
    jax_m1 = salt3nir_m1(phase, wave)
    assert jnp.isclose(jax_m1, snc_m1, atol=1e-5), f"Mismatch for M1 at phase {phase}, wave {wave}: SNCosmo {snc_m1}, JAX {jax_m1}"

# Update color law test to use the colorlaw method
@pytest.mark.parametrize("wave", [
    4000.0,
    5000.0,
    6000.0
])
def test_salt3nir_colorlaw(wave):
    snc_model = sncosmo.Model(source='salt3-nir')
    snc_cl = snc_model.source.colorlaw(np.array([wave]))
    jax_cl = salt3nir_colorlaw(wave)
    assert jnp.isclose(jax_cl, snc_cl, atol=1e-5), f"Mismatch for color law at wave {wave}: SNCosmo {snc_cl}, JAX {jax_cl}"

# Fix bandflux test to include zpsys parameter
@pytest.mark.parametrize("phase, wave, params", [
    (0.0, 4000.0, {'x0': 1.0, 'x1': 0.1, 'c': 0.0}),
    (10.0, 5000.0, {'x0': 1.0, 'x1': 0.1, 'c': 0.0}),
    (20.0, 6000.0, {'x0': 1.0, 'x1': 0.1, 'c': 0.0})
])
def test_salt3nir_bandflux(phase, wave, params):
    snc_model = sncosmo.Model(source='salt3-nir')
    snc_flux = snc_model.bandflux('bessellb', phase, params['x0'], zpsys='ab')
    jax_flux = salt3nir_flux(phase, wave, params)
    assert jnp.isclose(jax_flux, snc_flux, atol=1e-5), f"Mismatch for bandflux at phase {phase}, wave {wave}: SNCosmo {snc_flux}, JAX {jax_flux}"

class SALT3NIR_Model:
    def __init__(self, model_path=None):
        # Set default model path if not provided
        if model_path is None:
            model_path = 'sncosmo-modelfiles'
        
        # Model file paths
        model_dir = os.path.join(model_path, 'models/salt3-nir/salt3nir-p22')
        m0_file = os.path.join(model_dir, 'salt3_template_0.dat')
        m1_file = os.path.join(model_dir, 'salt3_template_1.dat')
        
        # Read data files
        phase_grid, wave_grid, m0_data = read_griddata_file(m0_file)
        _, _, m1_data = read_griddata_file(m1_file)
        
        # Store as instance variables
        self.phase_grid = phase_grid
        self.wave_grid = wave_grid
        self.m0_data = m0_data  # Scale factor already applied when loading data
        self.m1_data = m1_data  # Scale factor already applied when loading data
        
        # Model parameters
        self.params = {
            'x0': 1.0,
            'x1': 0.0,
            'c': 0.0,
            't0': 0.0
        }
    
    def set_param(self, param, value):
        """Set a model parameter value."""
        if param in self.params:
            self.params[param] = value
        else:
            raise ValueError(f"Unknown parameter: {param}")
    
    def salt3nir_m0(self, phase, wave):
        """Get the M0 component at the given phase and wavelength."""
        phase = jnp.asarray(phase).reshape(-1)  # Ensure 1D array
        wave = jnp.asarray(wave).reshape(-1)  # Ensure 1D array
        
        # Create a meshgrid of phase and wave values
        phase_grid_2d, wave_grid_2d = jnp.meshgrid(phase, wave, indexing='ij')
        
        # Use vmap to compute M0 values for all phase-wave combinations
        m0_values = jax.vmap(
            jax.vmap(
                lambda p, w: bicubic_interp2d(p, w, self.phase_grid, self.wave_grid, self.m0_data)
            )
        )(phase_grid_2d, wave_grid_2d)
        
        return m0_values
    
    def salt3nir_m1(self, phase, wave):
        """Get the M1 component at the given phase and wavelength."""
        phase = jnp.asarray(phase).reshape(-1)  # Ensure 1D array
        wave = jnp.asarray(wave).reshape(-1)  # Ensure 1D array
        
        # Create a meshgrid of phase and wave values
        phase_grid_2d, wave_grid_2d = jnp.meshgrid(phase, wave, indexing='ij')
        
        # Use vmap to compute M1 values for all phase-wave combinations
        m1_values = jax.vmap(
            jax.vmap(
                lambda p, w: bicubic_interp2d(p, w, self.phase_grid, self.wave_grid, self.m1_data)
            )
        )(phase_grid_2d, wave_grid_2d)
        
        return m1_values
    
    def salt3nir_colorlaw(self, wave):
        """Get the color law value at the given wavelength."""
        return salt3nir_colorlaw(wave)
    
    def flux(self, phase, wave):
        """Calculate the flux at the given phase and wavelength."""
        # Get the time-shifted phase
        phase_shifted = phase - self.params['t0']
        
        # Get the components
        m0 = self.salt3nir_m0(phase_shifted, wave)
        m1 = self.salt3nir_m1(phase_shifted, wave)
        cl = self.salt3nir_colorlaw(wave)
        
        # Calculate the flux
        return self.params['x0'] * (
            m0 + 
            self.params['x1'] * m1
        ) * jnp.exp(-self.params['c'] * cl * self.params['c'])

def salt3nir_model(params):
    """Create a SALT3-NIR model with the given parameters.
    
    Parameters
    ----------
    params : dict
        Dictionary containing model parameters:
        - x0 : float
            Overall flux normalization
        - x1 : float
            First component amplitude
        - c : float
            Color parameter
        - t0 : float
            Time of peak brightness
        - z : float
            Redshift
    
    Returns
    -------
    Model
        A Model instance configured with the SALT3-NIR source
    """
    from jax_supernovae.models import Model
    
    # Create model instance
    model = Model()
    
    # Set the flux function
    def flux_func(phase, wave):
        # Convert inputs to arrays
        phase = jnp.asarray(phase)
        wave = jnp.asarray(wave)
        
        # Convert to rest frame
        z = params.get('z', 0.0)
        t0 = params.get('t0', 0.0)
        a = 1.0 / (1.0 + z)
        
        # Get rest-frame phase and wavelength
        phase_rest = (phase - t0) * a
        wave_rest = wave * a
        
        # Reshape inputs for broadcasting
        if phase_rest.ndim == 0:
            phase_rest = phase_rest.reshape(1)
        if wave_rest.ndim == 0:
            wave_rest = wave_rest.reshape(1)
        
        # Get components
        m0 = salt3nir_m0(phase_rest, wave_rest)
        m1 = salt3nir_m1(phase_rest, wave_rest)
        cl = salt3nir_colorlaw(wave_rest)
        
        # Calculate flux (scale factor already applied to m0 and m1)
        flux = params['x0'] * (m0 + params['x1'] * m1) * 10**(-0.4 * cl * params['c'])
        
        # Return scalar if both inputs were scalar
        if phase.ndim == 0 and wave.ndim == 0:
            return flux[0]
        return flux
    
    # Set the model's flux function and parameters
    model.flux = flux_func
    model.parameters = params
    
    return model 

@partial(jax.jit, static_argnames=['bandpass'])
def salt3nir_bandflux(phase, bandpass, params):
    """Calculate bandflux for SALT3-NIR model.
    
    Parameters
    ----------
    phase : float or array_like
        Rest-frame phase in days relative to maximum brightness.
    bandpass : Bandpass object
        Bandpass to calculate flux through.
    params : dict
        Model parameters including z, t0, x0, x1, c.
        
    Returns
    -------
    float or array_like
        Flux in photons/s/cm^2. Return value is float if phase is scalar,
        array if phase is array.
    """
    # Check if phase is scalar
    is_scalar = jnp.ndim(phase) == 0
    if is_scalar:
        phase = jnp.array([phase])
    
    # Get parameters
    z = params['z']
    t0 = params['t0']
    x0 = params['x0']
    x1 = params['x1']
    c = params['c']
    
    # Convert to rest-frame phase
    a = 1.0 / (1.0 + z)  # Scale factor
    restphase = (phase - t0) * a
    
    # Set up wavelength grid
    wave, dwave = integration_grid(bandpass.minwave(), bandpass.maxwave(), MODEL_BANDFLUX_SPACING)
    restwave = wave * a
    
    # Get transmission and model components
    trans = bandpass(wave)
    m0 = salt3nir_m0(restphase[:, None], restwave[None, :])  # Shape: (n_phase, n_wave)
    m1 = salt3nir_m1(restphase[:, None], restwave[None, :])  # Shape: (n_phase, n_wave)
    cl = salt3nir_colorlaw(restwave)  # Shape: (n_wave,)
    
    # Calculate rest-frame flux
    rest_flux = x0 * (m0 + x1 * m1) * 10**(-0.4 * cl[None, :] * c) * a  # Shape: (n_phase, n_wave)
    
    # Integrate flux through bandpass using trapezoidal rule
    result = jnp.trapezoid(wave[None, :] * trans[None, :] * rest_flux, wave, axis=1) / HC_ERG_AA
    
    # Return scalar if input was scalar
    if is_scalar:
        result = result[0]
    
    return result 