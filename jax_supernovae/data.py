"""Data loading and processing utilities for JAX supernova models."""
import jax.numpy as jnp
import numpy as np
import os
from astropy.table import Table
from .bandpasses import register_all_bandpasses
import importlib.resources

# Get package directory
PACKAGE_DIR = os.path.dirname(__file__)

def find_object_filepath(base_dir, object_name):
    """Find the data file for a given object in the base directory.
    
    Parameters
    ----------
    base_dir : str
        Base directory to search in
    object_name : str
        Name of the object (e.g., '19agl')
        
    Returns
    -------
    str
        Full path to the data file
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
    """Load HSF data for a given object.
    
    Parameters
    ----------
    object_name : str
        Name of the object (e.g., '19agl')
    base_dir : str
        Base directory containing the data files. Defaults to 'data'.
        Expected structure is either:
        - [base_dir]/Ia/[object_name]/all.phot
        - Or any .dat/.phot file containing the object name
        
    Returns
    -------
    astropy.table.Table
        Table containing the processed data with columns:
        - time: observation times (from mjd)
        - band: filter/band names
        - flux: flux measurements
        - fluxerr: flux measurement errors
        - zp: zero points (defaults to 27.5 if not present)
            
    Raises
    ------
    FileNotFoundError
        If no data file is found for the given object
    ValueError
        If required columns are missing from the data file
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

def load_redshift(object_name, redshift_file='data/redshifts.dat', targets_file='data/targets.dat'):
    """Load redshift for a given object.
    
    First tries redshifts.dat (high-quality spectroscopic redshifts),
    then falls back to targets.dat if object not found.
    
    Parameters
    ----------
    object_name : str
        Name of the object (e.g., '19agl')
    redshift_file : str
        Path to redshifts.dat file
    targets_file : str
        Path to targets.dat file (fallback)
        
    Returns
    -------
    tuple
        (redshift, redshift_err, flag) where:
        - redshift is the heliocentric redshift
        - redshift_err is the symmetric error (max of plus/minus for redshifts.dat, 
          or 0.001 default for targets.dat)
        - flag is the reliability flag ('s'=strong, 'w'=weak, 'n'=no features,
          or 'spu' from targets.dat)
            
    Raises
    ------
    FileNotFoundError
        If neither redshift file nor targets file found
    ValueError
        If object not found in either file
    """
    # First try redshifts.dat (primary source)
    package_redshift_file = os.path.join(PACKAGE_DIR, 'data', 'redshifts.dat')
    if os.path.exists(package_redshift_file):
        redshift_file = package_redshift_file
        
    if os.path.exists(redshift_file):
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
        
        if measurements:
            # Prefer measurements with 's' flag, then 'w', then 'n'
            flag_priority = {'s': 0, 'w': 1, 'n': 2}
            measurements.sort(key=lambda x: flag_priority.get(x[3], 3))
            
            z, plus, minus, flag = measurements[0]
            z_err = max(plus, minus)
            
            return z, z_err, flag
    
    # Fallback to targets.dat
    package_targets_file = os.path.join(PACKAGE_DIR, 'data', 'targets.dat')
    if os.path.exists(package_targets_file):
        targets_file = package_targets_file
    elif not os.path.exists(targets_file):
        # Try in parent data directory
        parent_targets_file = os.path.join(os.path.dirname(PACKAGE_DIR), 'data', 'targets.dat')
        if os.path.exists(parent_targets_file):
            targets_file = parent_targets_file
        else:
            raise FileNotFoundError(f"Neither redshift file nor targets file found")
    
    # Read targets.dat
    with open(targets_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    data_lines = lines[1:] if lines and not lines[0].startswith('#') else lines
    
    for line in data_lines:
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 11 and parts[0].lower() == object_name.lower():
            try:
                # Account for classification field that may contain spaces (e.g., "SN Ia")
                # z_hel is typically at index 9 for entries with "SN Ia" classification
                # or index 8 for single-word classifications
                if len(parts) == 12:  # "SN Ia" case - two words
                    z_hel = parts[9]
                    z_flag = parts[11] if len(parts) > 11 else 'spu'
                elif len(parts) == 11:  # Single word classification
                    z_hel = parts[8]
                    z_flag = parts[10] if len(parts) > 10 else 'spu'
                else:
                    continue
                    
                if z_hel.lower() == 'none':
                    continue
                z = float(z_hel)
                z_err = 0.001  # Default error for targets.dat
                return z, z_err, z_flag
            except (ValueError, IndexError):
                continue
    
    raise ValueError(f"No redshift measurements found for object {object_name} in either redshifts.dat or targets.dat")

def load_and_process_data(sn_name, data_dir='data', fix_z=False):
    """Load and process supernova data, including bandpass registration and data array setup.
    
    Parameters
    ----------
    sn_name : str
        Name of the supernova to load (e.g., '19agl')
    data_dir : str
        Directory containing the data files. Defaults to 'data'.
    fix_z : bool
        Whether to fix redshift to value from redshifts.dat
        
    Returns
    -------
    tuple
        Contains processed data arrays and bridges:
        - times (jnp.array): Observation times
        - fluxes (jnp.array): Flux measurements
        - fluxerrs (jnp.array): Flux measurement errors
        - zps (jnp.array): Zero points
        - band_indices (jnp.array): Band indices
        - unique_bands (list): List of unique band names
        - bridges (tuple): Precomputed bridge data for each band
        - fixed_z (tuple or None): If fix_z is True, returns (z, z_err), else None
    """
    # Load data and register bandpasses
    data = load_hsf_data(sn_name, base_dir=data_dir)
    bandpass_dict, bridges_dict = register_all_bandpasses()

    # Get unique bands and their bridges
    unique_bands = []
    bridges = []
    for band in np.unique(data['band']):
        if band in bridges_dict:
            unique_bands.append(band)
            bridges.append(bridges_dict[band])
    # Convert bridges to tuple for JIT compatibility
    bridges = tuple(bridges)

    # Set up data arrays
    valid_mask = np.array([band in bandpass_dict for band in data['band']])
    times = jnp.array(data['time'][valid_mask])
    fluxes = jnp.array(data['flux'][valid_mask])
    fluxerrs = jnp.array(data['fluxerr'][valid_mask])
    zps = jnp.array(data['zp'][valid_mask])
    band_indices = jnp.array([unique_bands.index(band) for band in data['band'][valid_mask]])
    
    # Load redshift if requested
    fixed_z = None
    if fix_z:
        try:
            z, z_err, flag = load_redshift(sn_name)
            fixed_z = (z, z_err)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load redshift: {e}")
            fixed_z = None
    
    return times, fluxes, fluxerrs, zps, band_indices, unique_bands, bridges, fixed_z 

def get_all_supernovae_with_redshifts(redshift_file='data/redshifts.dat'):
    """Get all supernovae that have measured redshifts in redshifts.dat.
    
    Parameters
    ----------
    redshift_file : str
        Path to redshifts.dat file
        
    Returns
    -------
    list
        List of tuples (sn_name, z, z_err, flag) for all supernovae with redshifts
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

def load_multiple_supernovae(sn_names, data_dir='data', fix_z=False):
    """Load and process data for multiple supernovae with shared bandpass structure.
    
    Parameters
    ----------
    sn_names : list of str
        List of supernova names to load (e.g., ['19agl', '19dwz'])
    data_dir : str
        Directory containing the data files. Defaults to 'data'.
    fix_z : bool
        Whether to fix redshift to value from redshifts.dat
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'n_sne': Number of supernovae
        - 'sn_names': List of SN names
        - 'times_list': List of time arrays for each SN
        - 'fluxes_list': List of flux arrays for each SN
        - 'fluxerrs_list': List of flux error arrays for each SN
        - 'zps_list': List of zero point arrays for each SN
        - 'band_indices_list': List of band index arrays for each SN
        - 'sn_indices': Array mapping each observation to its SN index
        - 'all_times': Concatenated times for all SNe
        - 'all_fluxes': Concatenated fluxes for all SNe
        - 'all_fluxerrs': Concatenated flux errors for all SNe
        - 'all_zps': Concatenated zero points for all SNe
        - 'all_band_indices': Concatenated band indices for all SNe
        - 'unique_bands': List of unique band names across all SNe
        - 'bridges': Tuple of precomputed bridge data for unique bands
        - 'fixed_z_list': List of (z, z_err) tuples if fix_z=True, else None
        - 'n_bands': Number of unique bands
    """
    # Register all bandpasses once
    bandpass_dict, bridges_dict = register_all_bandpasses()
    
    # Collect all unique bands across all SNe first
    all_bands = set()
    for sn_name in sn_names:
        data = load_hsf_data(sn_name, base_dir=data_dir)
        valid_bands = [band for band in data['band'] if band in bandpass_dict]
        all_bands.update(valid_bands)
    
    # Create ordered list of unique bands
    unique_bands = sorted(list(all_bands))
    n_bands = len(unique_bands)
    
    # Create bridges for unique bands
    bridges = tuple([bridges_dict[band] for band in unique_bands])
    
    # Load data for each SN
    times_list = []
    fluxes_list = []
    fluxerrs_list = []
    zps_list = []
    band_indices_list = []
    fixed_z_list = [] if fix_z else None
    sn_indices_list = []
    
    for sn_idx, sn_name in enumerate(sn_names):
        # Load data for this SN
        data = load_hsf_data(sn_name, base_dir=data_dir)
        
        # Filter to valid bands and create indices into unique_bands
        valid_mask = np.array([band in bandpass_dict for band in data['band']])
        times = jnp.array(data['time'][valid_mask])
        fluxes = jnp.array(data['flux'][valid_mask])
        fluxerrs = jnp.array(data['fluxerr'][valid_mask])
        zps = jnp.array(data['zp'][valid_mask])
        
        # Map bands to indices in unique_bands list
        band_indices = jnp.array([unique_bands.index(band) for band in data['band'][valid_mask]])
        
        # Create SN index array
        sn_indices = jnp.full(len(times), sn_idx)
        
        # Store data
        times_list.append(times)
        fluxes_list.append(fluxes)
        fluxerrs_list.append(fluxerrs)
        zps_list.append(zps)
        band_indices_list.append(band_indices)
        sn_indices_list.append(sn_indices)
        
        # Load redshift if requested
        if fix_z:
            try:
                z, z_err, flag = load_redshift(sn_name)
                fixed_z_list.append((z, z_err))
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load redshift for {sn_name}: {e}")
                # Try to continue without fixing redshift for this SN
                raise ValueError(f"Cannot use fix_z=True when redshift is unavailable for {sn_name}. Either use fix_z=False or ensure all SNe have redshifts.")
    
    # Concatenate all data for efficient computation
    all_times = jnp.concatenate(times_list)
    all_fluxes = jnp.concatenate(fluxes_list)
    all_fluxerrs = jnp.concatenate(fluxerrs_list)
    all_zps = jnp.concatenate(zps_list)
    all_band_indices = jnp.concatenate(band_indices_list)
    all_sn_indices = jnp.concatenate(sn_indices_list)
    
    return {
        'n_sne': len(sn_names),
        'sn_names': sn_names,
        'times_list': times_list,
        'fluxes_list': fluxes_list,
        'fluxerrs_list': fluxerrs_list,
        'zps_list': zps_list,
        'band_indices_list': band_indices_list,
        'sn_indices': all_sn_indices,
        'all_times': all_times,
        'all_fluxes': all_fluxes,
        'all_fluxerrs': all_fluxerrs,
        'all_zps': all_zps,
        'all_band_indices': all_band_indices,
        'unique_bands': unique_bands,
        'bridges': bridges,
        'fixed_z_list': fixed_z_list,
        'n_bands': n_bands
    }