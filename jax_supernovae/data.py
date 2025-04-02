import jax.numpy as jnp
import numpy as np
import os
from astropy.table import Table
from .bandpasses import register_all_bandpasses, Bandpass # Import Bandpass if needed
import importlib.resources

# Get package directory
PACKAGE_DIR = os.path.dirname(__file__)

def find_object_filepath(base_dir, object_name):
    """
    Find the data file for a given object in the base directory.
    
    Args:
        base_dir (str): Base directory to search in
        object_name (str): Name of the object (e.g., '19agl')
        
    Returns:
        str: Full path to the data file
    """
    # First try direct path in object directory
    direct_path = os.path.join(base_dir, object_name, 'all.phot')
    if os.path.exists(direct_path):
        return direct_path
        
    # Then try path with Ia subdirectory
    ia_path = os.path.join(base_dir, 'Ia', object_name, 'all.phot')
    if os.path.exists(ia_path):
        return ia_path
        
    # If neither exists, do a recursive search
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if (object_name.lower() in file.lower() and 
                (file.endswith('.dat') or file.endswith('.phot'))):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No data file found for object {object_name}")

def load_hsf_data(object_name, base_dir='data'):
    """
    Load HSF data for a given object.
    
    Args:
        object_name (str): Name of the object (e.g., '19agl')
        base_dir (str): Base directory containing the data files. Defaults to 'data'.
                       Expected structure is either:
                       - [base_dir]/Ia/[object_name]/all.phot
                       - Or any .dat/.phot file containing the object name
        
    Returns:
        astropy.table.Table: Table containing the processed data with columns:
            - time: observation times (from mjd)
            - band: filter/band names
            - flux: flux measurements
            - fluxerr: flux measurement errors
            - zp: zero points (defaults to 27.5 if not present)
            
    Raises:
        FileNotFoundError: If no data file is found for the given object
        ValueError: If required columns are missing from the data file
    """
    # Try to find data in package data directory first
    package_data_dir = os.path.join(PACKAGE_DIR, 'data')
    try:
        data_file = find_object_filepath(package_data_dir, object_name)
    except FileNotFoundError:
        # If not found in package, try the user-provided directory
        data_file = find_object_filepath(base_dir, object_name)
    
    print(f"Loading data from {data_file}")

    # Read the data file
    data = Table.read(data_file, format='ascii')
    
    # Rename columns to match expected names
    if 'mjd' in data.colnames and 'time' not in data.colnames:
        data['time'] = data['mjd']
        data.remove_column('mjd')
    
    if 'bandpass' in data.colnames and 'band' not in data.colnames:
        data['band'] = data['bandpass']
        data.remove_column('bandpass')
    
    # Ensure required columns exist
    required_columns = {'time', 'band', 'flux', 'fluxerr'}
    missing_columns = required_columns - set(data.colnames)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Add zp column if not present (default to 27.5 as per common convention)
    if 'zp' not in data.colnames:
        data['zp'] = np.full(len(data), 27.5)
    
    # Sort by time
    data.sort('time')
    
    return data

def load_redshift(object_name, redshift_file='data/redshifts.dat'):
    """
    Load redshift for a given object from redshifts.dat.
    
    Args:
        object_name (str): Name of the object (e.g., '19agl')
        redshift_file (str): Path to redshifts.dat file
        
    Returns:
        tuple: (redshift, redshift_err, flag) where:
            - redshift is the heliocentric redshift
            - redshift_err is the symmetric error (max of plus/minus)
            - flag is the reliability flag ('s'=strong, 'w'=weak, 'n'=no features)
            
    Raises:
        FileNotFoundError: If redshift file not found
        ValueError: If object not found in redshift file
    """
    # Try package data directory first
    package_redshift_file = os.path.join(PACKAGE_DIR, 'data', 'redshifts.dat')
    if os.path.exists(package_redshift_file):
        redshift_file = package_redshift_file
    elif not os.path.exists(redshift_file):
        raise FileNotFoundError(f"Redshift file not found: {redshift_file}")
        
    # Skip comment lines and read data
    with open(redshift_file, 'r') as f:
        lines = f.readlines()
    
    data_lines = [l for l in lines if not l.startswith('#')]
    
    # Find all measurements for this object
    measurements = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        if parts[0].lower() == object_name.lower():
            try:
                z = float(parts[2])
                plus = float(parts[3])
                minus = float(parts[4])
                flag = parts[5] if len(parts) > 5 else 'n'
                measurements.append((z, plus, minus, flag))
            except (ValueError, IndexError):
                continue
    
    if not measurements:
        raise ValueError(f"No redshift measurements found for object {object_name}")
    
    # Prefer measurements with 's' flag, then 'w', then 'n'
    flag_priority = {'s': 0, 'w': 1, 'n': 2}
    measurements.sort(key=lambda x: flag_priority.get(x[3], 3))
    
    z, plus, minus, flag = measurements[0]
    z_err = max(plus, minus)
    
    return z, z_err, flag

def load_and_process_data(sn_name, data_dir='data', fix_z=False,
                          selected_bandpasses=None, custom_bandpass_files=None, svo_filters=None):
    """
    Load and process supernova data, including bandpass registration and data array setup.
    
    Args:
        sn_name (str): Name of the supernova to load (e.g., '19agl')
        data_dir (str): Directory containing the data files. Defaults to 'data'.
        fix_z (bool): Whether to fix redshift to value from redshifts.dat
        selected_bandpasses (list, optional): List of bandpass names to include. If None, uses all valid bands found in data.
        custom_bandpass_files (list or dict, optional): Passed to register_all_bandpasses.
        svo_filters (list, optional): Passed to register_all_bandpasses.
        
    Returns:
        tuple: Contains processed data arrays, bridges, zpbandfluxes, and metadata:
            - times (jnp.array): Observation times
            - fluxes (jnp.array): Flux measurements
            - fluxerrs (jnp.array): Flux measurement errors
            - zps (jnp.array): Zero points (per observation time)
            - band_indices (jnp.array): Indices mapping times to unique bands
            - bridges (tuple): Precomputed bridge data for each unique band used
            - zpbandfluxes (jnp.array): Precomputed AB zpbandflux for each unique band used
            - fixed_z (tuple or None): If fix_z is True, returns (z, z_err), else None
            - unique_bands (list): List of unique band names used (after filtering).
    """
    # Load raw photometry data
    data = load_hsf_data(sn_name, base_dir=data_dir)
    
    # Register all standard, custom, and SVO bandpasses
    # This returns dictionaries mapping *all* registered band names to their objects
    all_bandpass_dict, all_bridges_dict, all_zpbandfluxes_dict = register_all_bandpasses(
        custom_bandpass_files=custom_bandpass_files,
        svo_filters=svo_filters
    )

    # Determine unique bands present in the loaded data
    bands_in_data = np.unique(data['band'])
    
    # Filter bands based on selected_bandpasses if provided
    if selected_bandpasses:
        # Keep only bands that are both in the data AND in the selected list
        final_unique_bands = [b for b in bands_in_data if b in selected_bandpasses]
    else:
        # If no selection, use all bands found in the data
        final_unique_bands = list(bands_in_data)

    # Validate that all final selected bands are actually registered and have bridges/zpfluxes
    valid_final_bands = []
    final_bridges = []
    final_zpbandfluxes_list = []
    skipped_bands = set()

    for band in final_unique_bands:
        if band in all_bridges_dict and band in all_zpbandfluxes_dict:
            valid_final_bands.append(band)
            final_bridges.append(all_bridges_dict[band])
            final_zpbandfluxes_list.append(all_zpbandfluxes_dict[band])
        else:
            skipped_bands.add(band)
            print(f"Warning: Band '{band}' (selected or found in data) not found in registered bridges/zpbandfluxes. Skipping.")

    if not valid_final_bands:
         raise ValueError(f"No valid bandpasses found for SN {sn_name} after filtering and validation.")

    # Convert final bridges to tuple and zpbandfluxes to JAX array
    final_bridges = tuple(final_bridges)
    final_zpbandfluxes = jnp.array(final_zpbandfluxes_list)

    # Filter the original data table to include only points from the valid final bands
    data_mask = np.array([band in valid_final_bands for band in data['band']])
    
    if not np.any(data_mask):
        raise ValueError(f"No data points remaining for SN {sn_name} after filtering for valid bands: {valid_final_bands}")
        
    filtered_data = data[data_mask]

    # Create final JAX arrays from the filtered data
    times = jnp.array(filtered_data['time'])
    fluxes = jnp.array(filtered_data['flux'])
    fluxerrs = jnp.array(filtered_data['fluxerr'])
    zps = jnp.array(filtered_data['zp']) # zps are per time point
    
    # Map each time point in the filtered data to the index in the valid_final_bands list
    band_indices = jnp.array([valid_final_bands.index(band) for band in filtered_data['band']])
    
    # Load redshift if requested
    fixed_z = None
    if fix_z:
        try:
            z, z_err, flag = load_redshift(sn_name)
            fixed_z = (z, z_err)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load redshift for {sn_name}: {e}")
            fixed_z = None
            
    # Return all necessary arrays and the list of unique bands used
    return times, fluxes, fluxerrs, zps, band_indices, final_bridges, final_zpbandfluxes, fixed_z, valid_final_bands

def get_all_supernovae_with_redshifts(redshift_file='data/redshifts.dat'):
    """
    Get all supernovae that have measured redshifts in redshifts.dat.
    
    Args:
        redshift_file (str): Path to redshifts.dat file
        
    Returns:
        list: List of tuples (sn_name, z, z_err, flag) for all supernovae with redshifts
    """
    # Try package data directory first
    package_redshift_file = os.path.join(PACKAGE_DIR, 'data', 'redshifts.dat')
    if os.path.exists(package_redshift_file):
        redshift_file = package_redshift_file
    elif not os.path.exists(redshift_file):
        raise FileNotFoundError(f"Redshift file not found: {redshift_file}")
        
    # Skip comment lines and read data
    with open(redshift_file, 'r') as f:
        lines = f.readlines()
    
    data_lines = [l for l in lines if not l.startswith('#')]
    
    # Dictionary to store best measurements for each object
    best_measurements = {}
    
    # Flag priority (prefer strong features)
    flag_priority = {'s': 0, 'w': 1, 'n': 2}
    
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            sn_name = parts[0]
            z = float(parts[2])
            plus = float(parts[3])
            minus = float(parts[4])
            flag = parts[5] if len(parts) > 5 else 'n'
            
            # If we haven't seen this object before, or if this measurement has higher priority
            if (sn_name not in best_measurements or 
                flag_priority.get(flag, 3) < flag_priority.get(best_measurements[sn_name][3], 3)):
                best_measurements[sn_name] = (sn_name, z, max(plus, minus), flag)
        except (ValueError, IndexError):
            continue
    
    # Convert dictionary to sorted list
    return sorted(best_measurements.values())