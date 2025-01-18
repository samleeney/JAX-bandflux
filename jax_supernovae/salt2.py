"""SALT2 model data."""
import jax
import jax.numpy as jnp
import numpy as np
import sncosmo

# Get SALT2 source
salt2_source = sncosmo.get_source('salt2')

# Pre-compute the wavelength grid
wave_grid = jnp.array(salt2_source._wave)
phase_grid = jnp.array(salt2_source._phase)

# Pre-compute the template data by evaluating on the grid
# Note: Data is already scaled by 1e-12 in SNCosmo's initialization
M0_data = jnp.array(salt2_source._model['M0'](phase_grid, wave_grid))
M1_data = jnp.array(salt2_source._model['M1'](phase_grid, wave_grid))

# Pre-compute color law values on the wavelength grid
# Note: SALT2 color law requires float64 input
colorlaw_grid = jnp.array(salt2_source._colorlaw(np.array(wave_grid, dtype=np.float64)))

@jax.jit
def interp2d(x, y, xp, yp, zp):
    """2D linear interpolation in JAX."""
    # Find indices
    ix = jnp.searchsorted(xp, x)
    iy = jnp.searchsorted(yp, y)
    
    # Ensure we don't go out of bounds
    ix = jnp.clip(ix, 1, len(xp)-1)
    iy = jnp.clip(iy, 1, len(yp)-1)
    
    # Get surrounding points
    x0 = xp[ix-1]
    x1 = xp[ix]
    y0 = yp[iy-1]
    y1 = yp[iy]
    
    # Get values at corners
    z00 = zp[ix-1, iy-1]
    z10 = zp[ix, iy-1]
    z01 = zp[ix-1, iy]
    z11 = zp[ix, iy]
    
    # Compute weights
    wx = (x - x0) / (x1 - x0)
    wy = (y - y0) / (y1 - y0)
    
    # Interpolate
    return (1-wx)*(1-wy)*z00 + wx*(1-wy)*z10 + (1-wx)*wy*z01 + wx*wy*z11

@jax.jit
def salt2_m0(phase, wave):
    """JAX implementation of SALT2 M0 template interpolation."""
    return interp2d(phase, wave, phase_grid, wave_grid, M0_data)

@jax.jit
def salt2_m1(phase, wave):
    """JAX implementation of SALT2 M1 template interpolation."""
    return interp2d(phase, wave, phase_grid, wave_grid, M1_data)

@jax.jit
def salt2_colorlaw(wave):
    """JAX implementation of SALT2 color law."""
    # Interpolate pre-computed color law values
    return jnp.interp(wave, wave_grid, colorlaw_grid)

@jax.jit
def salt2_flux(t_rest, wave_rest, params):
    """Compute SALT2 model flux.
    
    Args:
        t_rest: Rest-frame time(s) in days relative to t0.
        wave_rest: Rest-frame wavelength(s) in Angstroms.
        params: Dictionary of model parameters.
        
    Returns:
        Flux in erg/s/cm^2/A.
    """
    # Reshape inputs for broadcasting
    t_rest = jnp.atleast_1d(t_rest)
    wave_rest = jnp.atleast_1d(wave_rest)
    
    # Get parameters
    x0 = params['x0']
    x1 = params['x1']
    c = params['c']
    
    # Interpolate M0 and M1 components
    m0 = interp2d(t_rest[:, None], wave_rest[None, :], phase_grid, wave_grid, M0_data)  # Shape: (ntime, nwave)
    m1 = interp2d(t_rest[:, None], wave_rest[None, :], phase_grid, wave_grid, M1_data)  # Shape: (ntime, nwave)
    
    # Get color law values at rest wavelengths
    colorlaw = jnp.interp(wave_rest, wave_grid, colorlaw_grid)  # Shape: (nwave,)
    
    # Return flux
    # Note: The flux is in rest frame
    return x0 * (m0 + x1 * m1) * 10**(-0.4 * colorlaw * c)