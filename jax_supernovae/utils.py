"""Utility functions for JAX supernova models."""
import numpy as np
import os
import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap
from jax_supernovae.constants import HC_ERG_AA

@partial(vmap, in_axes=(0, None, None))
def interp(x, xp, fp):
    """Linear interpolation for JAX arrays.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values
    xp : array_like
        The x-coordinates of the data points
    fp : array_like
        The y-coordinates of the data points
        
    Returns
    -------
    array_like
        The interpolated values
    """
    x = jnp.asarray(x)  # Don't reshape, preserve input shape
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    
    # Find indices of points to interpolate between
    i = jnp.searchsorted(xp, x)
    i = jnp.clip(i, 1, len(xp) - 1)
    
    # Get x and y values to interpolate between
    x0 = xp[i - 1]
    x1 = xp[i]
    y0 = fp[i - 1]
    y1 = fp[i]
    
    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


@jax.jit
def bandflux_integration(wave, trans, flux, dwave):
    """Generic bandflux integration for any source model.

    Computes synthetic photometry via: ∫ λ T(λ) F(λ) dλ / (hc)

    This is the core integration function used by all source models (SALT3,
    TimeSeriesSource, etc.). It integrates the product of wavelength,
    transmission, and flux over the bandpass wavelength range.

    Parameters
    ----------
    wave : jnp.array
        Integration wavelength grid (Angstroms). Shape (N,)
    trans : jnp.array
        Transmission values on wave grid (dimensionless). Shape (N,)
    flux : jnp.array
        Model flux on wave grid (erg/s/cm²/Å). Shape (N,) or (..., N) for vectorised
    dwave : float
        Grid spacing (Angstroms), typically 5.0 Å

    Returns
    -------
    float or jnp.array
        Integrated bandflux (photons/s/cm²)

    Notes
    -----
    The integration formula is:
        bandflux = ∫ λ T(λ) F(λ) dλ / (hc)

    where:
    - λ is wavelength
    - T(λ) is the bandpass transmission
    - F(λ) is the source spectral flux density
    - h is Planck's constant
    - c is the speed of light
    - hc = 1.9864458571489284e-08 erg·Å

    This matches sncosmo's integration method exactly (5.0 Å grid spacing).

    Examples
    --------
    ::

        wave = jnp.linspace(4000, 6000, 401)
        trans = jnp.ones_like(wave)
        flux = jnp.ones_like(wave) * 1e-15
        dwave = 5.0
        result = bandflux_integration(wave, trans, flux, dwave)
    """
    # Sum over wavelength dimension (last axis)
    # wave * trans * flux has shape (..., N)
    # sum over last axis gives (...,) or scalar
    return jnp.sum(wave * trans * flux, axis=-1) * dwave / HC_ERG_AA


@partial(jax.jit, static_argnames=['zpsys'])
def apply_zeropoint(flux, zp, zpsys):
    """Apply zero-point scaling to flux.

    Scales photon flux to match specified magnitude zero point system.

    Parameters
    ----------
    flux : float or jnp.array
        Model flux in photons/s/cm²
    zp : float, jnp.array, or None
        Zero point magnitude. If None, returns flux unchanged.
    zpsys : str
        Zero point system. Currently supports 'ab'.

    Returns
    -------
    float or jnp.array
        Scaled flux

    Notes
    -----
    Zero point scaling converts from native flux units to a specific
    photometric system. For AB magnitudes:
        flux_scaled = flux * 10^(0.4 * zp)

    This matches sncosmo's zero point handling.

    Examples
    --------
    ::

        flux = 1.0
        zp = 25.0
        scaled = apply_zeropoint(flux, zp, 'ab')
        # Returns: Array(10000., dtype=float64)
    """
    # Handle None case - return flux unchanged
    if zp is None:
        return flux

    # AB magnitude system scaling
    # mag = -2.5 * log10(flux) + zp
    # => flux_scaled = flux * 10^(0.4 * zp)
    if zpsys == 'ab':
        return flux * 10.0**(0.4 * zp)
    else:
        # For now, only AB system is implemented
        # Could add 'vega' and other systems in future
        raise ValueError(f"Zero point system '{zpsys}' not implemented. Only 'ab' is supported.")


def save_chains_dead_birth(dead_info, param_names=None, root_dir="chains"):
    """Save nested sampling results in dead-birth format without headers.

    Parameters
    ----------
    dead_info : NSInfo
        An object containing particles, logL, and logL_birth
    param_names : list, optional
        A list of parameter names
    root_dir : str, optional
        Directory to save chains in. Defaults to "chains".
        Will be created if it doesn't exist

    Notes
    -----
    The file contains `ndims + 2` columns in space-separated format:
    param1 param2 ... paramN logL logL_birth

    The file will be saved as [root_dir]/[root_dir]_dead-birth.txt
    """
    # Create directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    
    # Extract data from NSInfo
    points = np.array(dead_info.particles)
    logls_death = np.array(dead_info.loglikelihood)
    logls_birth = np.array(dead_info.loglikelihood_birth)
    
    # Combine data: parameters, death likelihood, birth likelihood
    data = np.column_stack([points, logls_death, logls_birth])
    
    # Construct output path
    output_path = os.path.join(root_dir, f"{root_dir}_dead-birth.txt")
    
    # Save without header
    np.savetxt(output_path, data)
    print(f"Saved {data.shape[0]} samples to {output_path}")