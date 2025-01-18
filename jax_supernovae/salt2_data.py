"""SALT2 model data."""
import jax.numpy as jnp
import sncosmo

# Get SALT2 source for wavelength grid
salt2_source = sncosmo.get_source('salt2')
wave_grid = jnp.array(salt2_source._wave)
phase_grid = jnp.array(salt2_source._phase)

def get_salt2_wave_grid():
    """Get the wavelength grid for SALT2 model."""
    return wave_grid

def get_salt2_phase_grid():
    """Get the phase grid for SALT2 model."""
    return phase_grid 