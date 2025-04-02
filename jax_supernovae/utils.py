import numpy as np
import os
import jax.numpy as jnp
from functools import partial
from jax import vmap

@partial(vmap, in_axes=(0, None, None))
def interp(x, xp, fp):
    """Linear interpolation for JAX arrays."""
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

def save_chains_dead_birth(dead_info, param_names=None, root_dir="chains", sn_name=None):
    """
    Save nested sampling results in dead-birth format without headers.

    Parameters:
    -----------
    dead_info : NSInfo
        An object containing particles, logL, and logL_birth.
    param_names : list, optional
        A list of parameter names.
    root_dir : str, optional
        Directory to save chains in. Defaults to "chains".
        Will be created if it doesn't exist.

    Notes:
    ------
    The file contains `ndims + 2` columns in space-separated format:
    param1 param2 ... paramN logL logL_birth

    If sn_name is provided, the file will be saved as
    [root_dir]/chains_[sn_name]_dead-birth.txt.
    Otherwise, it will be saved as [root_dir]/[root_dir]_dead-birth.txt.
    """
    # Create directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    
    # Extract data from NSInfo
    points = np.array(dead_info.particles)
    logls_death = np.array(dead_info.logL)
    logls_birth = np.array(dead_info.logL_birth)
    
    # Combine data: parameters, death likelihood, birth likelihood
    data = np.column_stack([points, logls_death, logls_birth])
    
    # Construct output path
    if sn_name:
        filename = f"chains_{sn_name}_dead-birth.txt"
    else:
        # Use basename of root_dir if no sn_name, handle potential edge case where root_dir is '.'
        base_name = os.path.basename(root_dir) if os.path.basename(root_dir) else 'chains'
        filename = f"{base_name}_dead-birth.txt"
    output_path = os.path.join(root_dir, filename)
    
    # Save without header
    np.savetxt(output_path, data)
    print(f"Saved {data.shape[0]} samples to {output_path}")