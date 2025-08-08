"""Utility functions for multi-supernova nested sampling with transmission calibration."""
import jax
import jax.numpy as jnp
import numpy as np
from jax_supernovae.data import load_and_process_data, load_hsf_data
from jax_supernovae.bandpasses import register_all_bandpasses

jax.config.update("jax_enable_x64", True)

def load_and_process_multiple_sne(sn_names, data_dir='data', fix_z=True):
    """Loads and processes data for multiple SNe, preparing it for JIT compilation.
    
    Parameters
    ----------
    sn_names : list of str
        List of supernova names to load
    data_dir : str
        Directory containing the data files
    fix_z : bool
        Whether to fix redshift
        
    Returns
    -------
    tuple
        (sne_data_jax, global_band_names, global_bridges) where:
        - sne_data_jax: PyTree of padded JAX arrays containing all data
        - global_band_names: List of unique band names across all SNe
        - global_bridges: Tuple of bridge data for each global band
    """
    # Register all standard bandpasses and get their bridges
    _, bridges_dict = register_all_bandpasses()
    
    # 1. First pass: Collect all data and determine global properties
    all_sn_data = []
    global_band_names = []
    max_n_points = 0
    max_n_bands_per_sn = 0
    all_sn_bands = []  # Store band names for each SN

    for sn_name in sn_names:
        # Load processed data
        data = load_and_process_data(sn_name, data_dir, fix_z)
        all_sn_data.append(data)
        
        # Load raw data to get band names
        phot_data = load_hsf_data(sn_name, base_dir=data_dir)
        sn_unique_bands = np.unique(phot_data['band']).tolist()
        all_sn_bands.append(sn_unique_bands)
        
        # Add to global band list
        for band in sn_unique_bands:
            if band not in global_band_names:
                global_band_names.append(band)
        
        # Track maximums for padding
        max_n_points = max(max_n_points, len(data[0]))
        max_n_bands_per_sn = max(max_n_bands_per_sn, len(sn_unique_bands))
    
    # Sort global band names for consistency
    global_band_names = sorted(global_band_names)
    global_band_map = {name: i for i, name in enumerate(global_band_names)}
    
    # Create tuple of bridges in global band order
    global_bridges = []
    for name in global_band_names:
        if name in bridges_dict:
            global_bridges.append(bridges_dict[name])
        else:
            print(f"Warning: Bridge not found for band {name}")
    global_bridges = tuple(global_bridges)
    
    n_sne = len(sn_names)
    
    # 2. Second pass: Create padded arrays
    sne_data = {
        'times': np.full((n_sne, max_n_points), -1.0),
        'fluxes': np.zeros((n_sne, max_n_points)),
        'fluxerrs': np.full((n_sne, max_n_points), 1e10),  # Large error for invalid points
        'zps': np.zeros((n_sne, max_n_points)),
        'band_indices': np.full((n_sne, max_n_points), -1, dtype=int),
        'valid_mask': np.zeros((n_sne, max_n_points), dtype=bool),
        'local_to_global_map': np.full((n_sne, max_n_bands_per_sn), -1, dtype=int),
        'fixed_z': np.zeros(n_sne),
        'n_points': np.zeros(n_sne, dtype=int),
        'n_bands': np.zeros(n_sne, dtype=int),
        'sn_bridges': []  # Will store per-SN bridge tuples
    }
    
    for i, sn_name in enumerate(sn_names):
        times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = all_sn_data[i]
        n_p = len(times)
        
        # Map local bands to global band indices first to know which are valid
        sn_unique_bands = all_sn_bands[i]
        
        # Create mapping of original band index to new compressed index (only valid bands)
        original_to_compressed = {}
        compressed_idx = 0
        valid_bands = []
        
        for j, band_name in enumerate(sn_unique_bands):
            if band_name in global_band_map and global_band_map[band_name] < len(global_bridges):
                original_to_compressed[j] = compressed_idx
                valid_bands.append(band_name)
                compressed_idx += 1
            else:
                original_to_compressed[j] = -1  # Invalid band
        
        # Remap band_indices to compressed indices
        # Convert to int if it's a JAX array
        if hasattr(band_indices, 'item'):
            band_indices_int = [int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in band_indices]
        else:
            band_indices_int = [int(idx) for idx in band_indices]
        remapped_band_indices = np.array([original_to_compressed.get(idx, -1) for idx in band_indices_int])
        
        # Filter out data points with invalid bands
        valid_points = remapped_band_indices >= 0
        n_valid = np.sum(valid_points)
        
        # Fill data arrays with only valid points
        if n_valid > 0:
            valid_times = times[valid_points]
            valid_fluxes = fluxes[valid_points]
            valid_fluxerrs = fluxerrs[valid_points]
            valid_zps = zps[valid_points]
            valid_remapped_indices = remapped_band_indices[valid_points]
            
            sne_data['times'][i, :n_valid] = valid_times
            sne_data['fluxes'][i, :n_valid] = valid_fluxes
            sne_data['fluxerrs'][i, :n_valid] = valid_fluxerrs
            sne_data['zps'][i, :n_valid] = valid_zps
            sne_data['band_indices'][i, :n_valid] = valid_remapped_indices
            sne_data['valid_mask'][i, :n_valid] = True
            sne_data['n_points'][i] = n_valid
        else:
            sne_data['n_points'][i] = 0
            
        sne_data['fixed_z'][i] = fixed_z[0] if fixed_z is not None and len(fixed_z) > 0 else 0.01
        sne_data['n_bands'][i] = len(valid_bands)
        
        # Create local to global mapping and collect bridges for valid bands only
        sn_bridges = []
        for j, band_name in enumerate(valid_bands):
            global_idx = global_band_map[band_name]
            sne_data['local_to_global_map'][i, j] = global_idx
            sn_bridges.append(global_bridges[global_idx])
        
        # Store bridges for this SN
        if len(sn_bridges) == 0:
            print(f"Warning: SN {sn_names[i]} has no valid bridges after filtering")
            sne_data['sn_bridges'].append(tuple())
        else:
            sne_data['sn_bridges'].append(tuple(sn_bridges))
    
    # Convert numpy arrays to JAX arrays (except sn_bridges which stays as tuple)
    sne_data_jax = {}
    for key, val in sne_data.items():
        if key != 'sn_bridges':
            sne_data_jax[key] = jnp.array(val)
        else:
            sne_data_jax[key] = val
    
    return sne_data_jax, global_band_names, global_bridges


def unpack_parameters(params, n_bands_global, n_sne, n_params_sn=4):
    """Unpack flat parameter vector into transmission shifts and SN parameters.
    
    Parameters
    ----------
    params : jnp.array
        Flat parameter vector
    n_bands_global : int
        Number of global bands (transmission shift parameters)
    n_sne : int
        Number of supernovae
    n_params_sn : int
        Number of parameters per supernova
        
    Returns
    -------
    tuple
        (global_shifts, sn_params) where:
        - global_shifts: Array of transmission shifts for each band
        - sn_params: 2D array of SN parameters (n_sne x n_params_sn)
    """
    global_shifts = params[:n_bands_global]
    sn_params = params[n_bands_global:].reshape(n_sne, n_params_sn)
    return global_shifts, sn_params


def pack_parameters(global_shifts, sn_params):
    """Pack transmission shifts and SN parameters into flat vector.
    
    Parameters
    ----------
    global_shifts : array
        Transmission shifts for each band
    sn_params : 2D array
        SN parameters (n_sne x n_params_sn)
        
    Returns
    -------
    jnp.array
        Flat parameter vector
    """
    return jnp.concatenate([global_shifts, sn_params.flatten()])