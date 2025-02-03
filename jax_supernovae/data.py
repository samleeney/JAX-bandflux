import jax.numpy as jnp
import numpy as np
from .bandpasses import register_all_bandpasses
from load_hsf_data import load_hsf_data

def load_and_process_data(sn_name):
    """
    Load and process supernova data, including bandpass registration and data array setup.
    
    Args:
        sn_name (str): Name of the supernova to load (e.g., '19agl')
        
    Returns:
        tuple: Contains processed data arrays and bridges:
            - times (jnp.array): Observation times
            - fluxes (jnp.array): Flux measurements
            - fluxerrs (jnp.array): Flux measurement errors
            - zps (jnp.array): Zero points
            - band_indices (jnp.array): Band indices
            - bridges (tuple): Precomputed bridge data for each band
    """
    # Load data and register bandpasses
    data = load_hsf_data(sn_name)
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

    return times, fluxes, fluxerrs, zps, band_indices, bridges 