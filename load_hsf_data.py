import os
import numpy as np
from astropy.table import Table

def find_object_filepath(base_dir, object_name):
    """
    Find the data file for a given object in the base directory.
    
    Args:
        base_dir (str): Base directory to search in
        object_name (str): Name of the object (e.g., '19agl')
        
    Returns:
        str: Full path to the data file
    """
    # First try direct path for known structure
    direct_path = os.path.join(base_dir, 'Ia', object_name, 'all.phot')
    if os.path.exists(direct_path):
        return direct_path
        
    # If direct path doesn't exist, do a recursive search
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if (object_name.lower() in file.lower() and 
                (file.endswith('.dat') or file.endswith('.phot'))):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No data file found for object {object_name}")

def load_hsf_data(object_name, base_dir='hsf_DR1'):
    """
    Load HSF data for a given object.
    
    Args:
        object_name (str): Name of the object (e.g., '19agl')
        base_dir (str): Base directory containing the data files
        
    Returns:
        astropy.table.Table: Table containing the processed data with columns:
            - time: observation times (from mjd)
            - band: filter/band names (from bandpass)
            - flux: flux measurements
            - fluxerr: flux measurement errors
            - zp: zero points
    """
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