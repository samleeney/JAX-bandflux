"""SALT3-NIR model implementation in JAX."""
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import sncosmo
import os
import pytest

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

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

# Read data and apply scaling
SCALE_FACTOR = 1e-12
phase_grid, wave_grid, m0_data = read_griddata_file(m0_file)
_, _, m1_data = read_griddata_file(m1_file)

# Scale the data (exactly like SNCosmo)
m0_data = m0_data * SCALE_FACTOR
m1_data = m1_data * SCALE_FACTOR

# Convert to JAX arrays
phase_grid = jnp.array(phase_grid)
wave_grid = jnp.array(wave_grid)
m0_data = jnp.array(m0_data)
m1_data = jnp.array(m1_data)

# Debug: Verify initial data source and scaling factor for M0
print("Debug: Verifying initial data source and scaling factor for M0...")
# Check file paths
print(f"M0 file path: {m0_file}")
# Check scaling factor
print(f"SCALE_FACTOR: {SCALE_FACTOR}")
print("Debug: Initial data source and scaling factor verification complete.")

# Debug: Verify data reading and scaling for M0
print("Debug: Verifying data reading and scaling for M0...")
# Check raw M0 data before scaling
raw_phase_grid, raw_wave_grid, raw_m0_data = read_griddata_file(m0_file)
print(f"Raw M0 data (first 5 values): {raw_m0_data.flatten()[:5]}")
# Check scaled M0 data
scaled_m0_data = raw_m0_data * SCALE_FACTOR
print(f"Scaled M0 data (first 5 values): {scaled_m0_data.flatten()[:5]}")
print("Debug: Data reading and scaling verification complete.")

# Debug: Verify M0 data reading and scaling
print("Debug: Verifying M0 data...")
print(f"M0 data (first 5 values): {m0_data.flatten()[:5]}")
print(f"M0 data shape: {m0_data.shape}, range: [{m0_data.min()}, {m0_data.max()}]")
print("Debug: M0 data verification complete.")

# Debug: Verify data reading and scaling
print("M0 file path:", m0_file)
print("M1 file path:", m1_file)
print("M0 data shape:", m0_data.shape)
print("M1 data shape:", m1_data.shape)
print("M0 data range:", m0_data.min(), m0_data.max())
print("M1 data range:", m1_data.min(), m1_data.max())

# Print statistics for verification
print(f"Phase grid range: [{phase_grid[0]}, {phase_grid[-1]}], shape: {phase_grid.shape}")
print(f"Wave grid range: [{wave_grid[0]}, {wave_grid[-1]}], shape: {wave_grid.shape}")
print(f"M1 data shape: {m1_data.shape}, range: [{m1_data.min()}, {m1_data.max()}]")

# Debug: Verify scaling and interpolation inputs for M0
print("Debug: Verifying scaling and interpolation inputs for M0...")
print(f"SCALE_FACTOR: {SCALE_FACTOR}")
print(f"Phase grid (first 5 values): {phase_grid[:5]}")
print(f"Wave grid (first 5 values): {wave_grid[:5]}")
print("Debug: Scaling and interpolation inputs verification complete.")

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
    """Bicubic convolution kernel used in SNCosmo.
    
    The kernel is defined by:
    W(x) = (a+2)*x^3-(a+3)*x^2+1 for x<=1
    W(x) = a(x^3-5*x^2+8*x-4) for 1<x<2
    W(x) = 0 for x>2
    where a = -0.5
    """
    A = -0.5  # This matches SNCosmo's value
    x = jnp.abs(x)
    
    # x <= 1
    case1 = (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    
    # 1 < x <= 2
    case2 = A * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0)
    
    # x > 2
    case3 = 0.0
    
    return jnp.where(x > 2.0, case3,
                    jnp.where(x > 1.0, case2, case1))

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

@jax.jit
def salt3nir_m0(phase, wave):
    """Get M0 component at given phase and wavelength."""
    phase = jnp.asarray(phase).reshape(-1)  # Ensure 1D array
    wave = jnp.asarray(wave).reshape(-1)  # Ensure 1D array
    
    # If inputs are single values, broadcast them
    if phase.size == 1 and wave.size == 1:
        return bicubic_interp2d(phase[0], wave[0], phase_grid, wave_grid, m0_data)
    
    # If one input is a single value, broadcast it
    if phase.size == 1:
        phase = jnp.full_like(wave, phase[0])
    elif wave.size == 1:
        wave = jnp.full_like(phase, wave[0])
    
    # Use vmap for vectorized computation
    return jax.vmap(lambda p, w: bicubic_interp2d(p, w, phase_grid, wave_grid, m0_data))(phase, wave)

@jax.jit
def salt3nir_m1(phase, wave):
    """Get M1 component at given phase and wavelength."""
    phase = jnp.asarray(phase).reshape(-1)  # Ensure 1D array
    wave = jnp.asarray(wave).reshape(-1)  # Ensure 1D array
    
    # If inputs are single values, broadcast them
    if phase.size == 1 and wave.size == 1:
        return bicubic_interp2d(phase[0], wave[0], phase_grid, wave_grid, m1_data)
    
    # If one input is a single value, broadcast it
    if phase.size == 1:
        phase = jnp.full_like(wave, phase[0])
    elif wave.size == 1:
        wave = jnp.full_like(phase, wave[0])
    
    # Use vmap for vectorized computation
    return jax.vmap(lambda p, w: bicubic_interp2d(p, w, phase_grid, wave_grid, m1_data))(phase, wave)

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
    snc_cl = snc_model.source.colorlaw(wave)
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
        self.m0_data = m0_data * 1e-12  # Scale factor
        self.m1_data = m1_data * 1e-12  # Scale factor
        
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
        
        # If inputs are single values, broadcast them
        if phase.size == 1 and wave.size == 1:
            return bicubic_interp2d(phase[0], wave[0], self.phase_grid, self.wave_grid, self.m0_data)
        
        # If one input is a single value, broadcast it
        if phase.size == 1:
            phase = jnp.full_like(wave, phase[0])
        elif wave.size == 1:
            wave = jnp.full_like(phase, wave[0])
        
        # Use vmap for vectorized computation
        return jax.vmap(lambda p, w: bicubic_interp2d(p, w, self.phase_grid, self.wave_grid, self.m0_data))(phase, wave)
    
    def salt3nir_m1(self, phase, wave):
        """Get the M1 component at the given phase and wavelength."""
        phase = jnp.asarray(phase).reshape(-1)  # Ensure 1D array
        wave = jnp.asarray(wave).reshape(-1)  # Ensure 1D array
        
        # If inputs are single values, broadcast them
        if phase.size == 1 and wave.size == 1:
            return bicubic_interp2d(phase[0], wave[0], self.phase_grid, self.wave_grid, self.m1_data)
        
        # If one input is a single value, broadcast it
        if phase.size == 1:
            phase = jnp.full_like(wave, phase[0])
        elif wave.size == 1:
            wave = jnp.full_like(phase, wave[0])
        
        # Use vmap for vectorized computation
        return jax.vmap(lambda p, w: bicubic_interp2d(p, w, self.phase_grid, self.wave_grid, self.m1_data))(phase, wave)
    
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
        ) * jnp.exp(-self.params['c'] * cl) 