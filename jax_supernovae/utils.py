import numpy as np
import os

def save_chains_dead_birth(dead_info, param_names=None, root_dir="chains"):
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

    The file will be saved as [root_dir]/[root_dir]_dead-birth.txt
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
    output_path = os.path.join(root_dir, f"{root_dir}_dead-birth.txt")
    
    # Save without header
    np.savetxt(output_path, data)
    print(f"Saved {data.shape[0]} samples to {output_path}")

# Example usage (add after your nested sampling run):
# param_names = [f'x{i}' for i in range(d)]  # d is your dimension from jax_ex.py
# save_getdist_chains(dead, logw, param_names) 