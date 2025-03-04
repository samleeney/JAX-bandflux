import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
import os
import yaml
import pandas as pd
from blackjax.ns.utils import log_weights
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
from jax_supernovae.bandpasses import register_bandpass, get_bandpass, register_all_bandpasses, Bandpass
from jax_supernovae.utils import save_chains_dead_birth
from jax_supernovae.data import load_redshift, load_hsf_data
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes
import requests

# Define default settings for nested sampling and prior bounds
DEFAULT_NS_SETTINGS = {
    'max_iterations': int(os.environ.get('NS_MAX_ITERATIONS', '500')),
    'n_delete': 75,
    'n_live': 150,
    'num_mcmc_steps_multiplier': 5,
    'fit_sigma': False,
    'fit_log_p': True,
    'fit_z': True
}

DEFAULT_PRIOR_BOUNDS = {
    'z': {'min': 0.001, 'max': 0.2},
    't0': {'min': 58000.0, 'max': 60000.0},
    'x0': {'min': -5.0, 'max': -1},
    'x1': {'min': -10, 'max': 10},
    'c': {'min': -0.6, 'max': 0.6},
    'sigma': {'min': 0.001, 'max': 5},
    'log_p': {'min': -20, 'max': -1}
}

# Default settings
DEFAULT_SETTINGS = {
    'fix_z': True,
    'sn_name': '19vnk',  # Default supernova to analyze
    'selected_bandpasses': None,  # Default: use all available bandpasses
    'custom_bandpass_files': None  # Default: no custom bandpass files
}

# Try to load settings.yaml; if not found, use an empty dictionary
try:
    with open('settings.yaml', 'r') as f:
        settings_from_file = yaml.safe_load(f)
except FileNotFoundError:
    settings_from_file = {}

# Merge the settings from file with the defaults
settings = DEFAULT_SETTINGS.copy()
settings.update(settings_from_file)

fix_z = settings['fix_z']
sn_name = settings['sn_name']
selected_bandpasses = settings.get('selected_bandpasses', None)
custom_bandpass_files = settings.get('custom_bandpass_files', None)

NS_SETTINGS = DEFAULT_NS_SETTINGS.copy()
NS_SETTINGS.update(settings.get('nested_sampling', {}))

PRIOR_BOUNDS = DEFAULT_PRIOR_BOUNDS.copy()
if 'prior_bounds' in settings:
    PRIOR_BOUNDS.update(settings['prior_bounds'])

# Option flag: when fit_sigma is True, an extra parameter is added
fit_sigma = NS_SETTINGS['fit_sigma']

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

def create_wfcam_j_bandpass():
    """
    Create a bandpass object for the WFCAM J filter.
    
    This function attempts to load the WFCAM J filter profile from the SVO Filter Profile Service.
    The filter profile must be downloaded first using the download_svo_filter.py script.
    
    Returns:
        Bandpass: A Bandpass object for the WFCAM J filter.
        
    Raises:
        FileNotFoundError: If the WFCAM J filter profile file is not found.
    """
    from jax_supernovae.bandpasses import Bandpass
    import os
    import numpy as np
    
    # Define possible paths to the WFCAM J filter profile
    filter_paths = [
        # Check in the filter_data directory (where download_svo_filter.py saves it)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'filter_data', 'UKIRT_WFCAM.J.dat'),
        
        # Check in the current directory
        'WFCAM_J.dat',
        'UKIRT_WFCAM.J.dat',
        
        # Check in the examples directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WFCAM_J.dat'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UKIRT_WFCAM.J.dat'),
        
        # Check in the filter_data directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../filter_data/UKIRT_WFCAM.J.dat'),
    ]
    
    # Try each path
    for filter_path in filter_paths:
        if os.path.exists(filter_path):
            try:
                print(f"Loading WFCAM J filter from {filter_path}")
                data = np.loadtxt(filter_path)
                wave = data[:, 0]  # Wavelength in Angstroms
                trans = data[:, 1]  # Transmission
                
                # Create and return the bandpass object
                return Bandpass(wave, trans)
            except Exception as e:
                print(f"Failed to load filter from {filter_path}: {e}")
    
    # If we get here, we couldn't find the filter file
    raise FileNotFoundError(
        "WFCAM J filter profile file not found. "
        "Please run the download_svo_filter.py script to download it."
    )

def custom_load_and_process_data(sn_name, data_dir='data', fix_z=False, selected_bandpasses=None, custom_bandpass_files=None):
    """
    Custom version of load_and_process_data that allows filtering by selected bandpasses.
    
    Args:
        sn_name (str): Name of the supernova to load (e.g., '19agl')
        data_dir (str): Directory containing the data files. Defaults to 'data'.
        fix_z (bool): Whether to fix redshift to value from redshifts.dat
        selected_bandpasses (list): List of bandpass names to include (e.g., ['g', 'r', 'i'])
                                   If None, all available bandpasses are used.
        custom_bandpass_files (list or dict): List of file paths to custom bandpass files,
                                             or a dictionary mapping bandpass names to file paths.
        
    Returns:
        tuple: Contains processed data arrays and bridges:
            - times (jnp.array): Observation times
            - fluxes (jnp.array): Flux measurements
            - fluxerrs (jnp.array): Flux measurement errors
            - zps (jnp.array): Zero points
            - band_indices (jnp.array): Band indices
            - bridges (tuple): Precomputed bridge data for each band
            - fixed_z (tuple or None): If fix_z is True, returns (z, z_err), else None
    """
    from jax_supernovae.data import load_hsf_data, load_redshift
    from jax_supernovae.salt3 import precompute_bandflux_bridge
    from jax_supernovae.bandpasses import register_bandpass
    import os
    
    # Load data
    data = load_hsf_data(sn_name, base_dir=data_dir)
    
    # Define J-band variants - only include the ones we actually need
    j_variants = ['J', 'J_1D3', 'J_2D']
    
    # Keep track of the original selected bandpasses (before any automatic additions)
    original_selected_bandpasses = None
    if selected_bandpasses is not None:
        original_selected_bandpasses = selected_bandpasses.copy() if isinstance(selected_bandpasses, list) else [selected_bandpasses]
    
    # Register bandpasses, including custom ones if provided
    try:
        # Try with the new function signature that accepts custom_bandpass_files
        bandpass_dict, bridges_dict = register_all_bandpasses(custom_bandpass_files)
        
        # Find all J-band variants in the data
        data_j_variants = [band for band in np.unique(data['band']) if band in j_variants]
        
        # Find all J-band variants in selected_bandpasses
        selected_j_variants = []
        if selected_bandpasses is not None:
            selected_j_variants = [band for band in selected_bandpasses if band in j_variants]
        
        # Determine if we need to register the J bandpass
        j_bandpass_needed = (len(data_j_variants) > 0 or len(selected_j_variants) > 0) and 'J' not in bridges_dict
        
        if j_bandpass_needed:
            print("J-band variant(s) found in data or selected bandpasses. Attempting to register J bandpass...")
            try:
                # Create the J bandpass
                j_bandpass = create_wfcam_j_bandpass()
                
                # Register the standard J bandpass if needed
                if 'J' in selected_j_variants or 'J' in data_j_variants:
                    register_bandpass('J', j_bandpass, force=True)
                    bandpass_dict['J'] = j_bandpass
                    bridges_dict['J'] = precompute_bandflux_bridge(j_bandpass)
                    print("Successfully registered WFCAM J bandpass")
                
                # Register all other J-band variants found in the data or selected bandpasses
                all_needed_variants = list(set(data_j_variants + selected_j_variants))
                for variant in all_needed_variants:
                    if variant != 'J' and variant not in bridges_dict:
                        register_bandpass(variant, j_bandpass, force=True)
                        bandpass_dict[variant] = j_bandpass
                        bridges_dict[variant] = precompute_bandflux_bridge(j_bandpass)
                        print(f"Successfully registered {variant} bandpass (using standard J filter profile)")
                
                # If any J-band variant is in the data but not in selected_bandpasses, add it
                if selected_bandpasses is not None:
                    for variant in data_j_variants:
                        if variant not in selected_bandpasses:
                            selected_bandpasses.append(variant)
                            print(f"Added {variant} to selected_bandpasses because it exists in the data")
            
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run the download_svo_filter.py script to download the WFCAM J filter profile.")
                print("Removing all J-band variants from selected_bandpasses.")
                if selected_bandpasses is not None:
                    selected_bandpasses = [b for b in selected_bandpasses if b not in j_variants]
            except Exception as e:
                print(f"Warning: Failed to register WFCAM J bandpass: {e}")
                # If we can't register the bandpasses, remove them from selected_bandpasses
                if selected_bandpasses is not None:
                    selected_bandpasses = [b for b in selected_bandpasses if b not in j_variants]
                    print("Removed all J-band variants from selected_bandpasses due to registration failure")
    except TypeError:
        # Fall back to the old function signature
        print("Using original register_all_bandpasses function (without custom bandpass support)")
        bandpass_dict, bridges_dict = register_all_bandpasses()
        
        # If custom bandpass files were provided, warn the user
        if custom_bandpass_files:
            print("Warning: Custom bandpass files were provided but are not supported by this version of the code.")
            print("Please update jax_supernovae/bandpasses.py to support custom bandpasses.")
            
        # Check for J-band variants in the data or selected bandpasses
        j_variants = ['J', 'J_1D3', 'J_2D']
        
        # Find all J-band variants in the data
        data_j_variants = [band for band in np.unique(data['band']) if band in j_variants]
        
        # Find all J-band variants in selected_bandpasses
        selected_j_variants = []
        if selected_bandpasses is not None:
            selected_j_variants = [band for band in selected_bandpasses if band in j_variants]
        
        # Determine if we need to register the J bandpass
        j_bandpass_needed = (len(data_j_variants) > 0 or len(selected_j_variants) > 0) and 'J' not in bridges_dict
        
        if j_bandpass_needed:
            print("J-band variant(s) found in data or selected bandpasses. Attempting to register J bandpass...")
            try:
                # Create the J bandpass
                j_bandpass = create_wfcam_j_bandpass()
                
                # Register the standard J bandpass if needed
                if 'J' in selected_j_variants or 'J' in data_j_variants:
                    register_bandpass('J', j_bandpass, force=True)
                    bandpass_dict['J'] = j_bandpass
                    bridges_dict['J'] = precompute_bandflux_bridge(j_bandpass)
                    print("Successfully registered WFCAM J bandpass")
                
                # Register all other J-band variants found in the data or selected bandpasses
                all_needed_variants = list(set(data_j_variants + selected_j_variants))
                for variant in all_needed_variants:
                    if variant != 'J' and variant not in bridges_dict:
                        register_bandpass(variant, j_bandpass, force=True)
                        bandpass_dict[variant] = j_bandpass
                        bridges_dict[variant] = precompute_bandflux_bridge(j_bandpass)
                        print(f"Successfully registered {variant} bandpass (using standard J filter profile)")
                
                # If any J-band variant is in the data but not in selected_bandpasses, add it
                if selected_bandpasses is not None:
                    for variant in data_j_variants:
                        if variant not in selected_bandpasses:
                            selected_bandpasses.append(variant)
                            print(f"Added {variant} to selected_bandpasses because it exists in the data")
            
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run the download_svo_filter.py script to download the WFCAM J filter profile.")
                print("Removing all J-band variants from selected_bandpasses.")
                if selected_bandpasses is not None:
                    selected_bandpasses = [b for b in selected_bandpasses if b not in j_variants]
            except Exception as e:
                print(f"Warning: Failed to register WFCAM J bandpass: {e}")
                # If we can't register the bandpasses, remove them from selected_bandpasses
                if selected_bandpasses is not None:
                    selected_bandpasses = [b for b in selected_bandpasses if b not in j_variants]
                    print("Removed all J-band variants from selected_bandpasses due to registration failure")
    
    # Print summary of J-band data
    for variant in j_variants:
        variant_count = np.sum(data['band'] == variant)
        if variant_count > 0:
            print(f"Found {variant_count} {variant} data points in the dataset.")
    
    # Filter by selected bandpasses if specified
    if selected_bandpasses is not None:
        # Convert to list if a single string was provided
        if isinstance(selected_bandpasses, str):
            selected_bandpasses = [selected_bandpasses]
            
        # Validate that all requested bandpasses exist
        for band in selected_bandpasses:
            if band not in bridges_dict:
                available_bands = list(bridges_dict.keys())
                raise ValueError(f"Bandpass '{band}' not found. Available bandpasses: {available_bands}")
        
        # Get unique bands and their bridges, filtered by selected_bandpasses
        unique_bands = []
        bridges = []
        for band in np.unique(data['band']):
            if band in bridges_dict and (selected_bandpasses is None or band in selected_bandpasses):
                unique_bands.append(band)
                bridges.append(bridges_dict[band])
        
        # Add selected bandpasses that aren't in the data, but only if they were in the original selection
        if original_selected_bandpasses:
            for band in original_selected_bandpasses:
                if band not in unique_bands and band in bridges_dict:
                    unique_bands.append(band)
                    bridges.append(bridges_dict[band])
                    print(f"Added {band} band to the list of unique bands (explicitly requested)")
    else:
        # Get all unique bands and their bridges
        unique_bands = []
        bridges = []
        for band in np.unique(data['band']):
            if band in bridges_dict:
                unique_bands.append(band)
                bridges.append(bridges_dict[band])
    
    # Convert bridges to tuple for JIT compatibility
    bridges = tuple(bridges)

    # Set up data arrays, filtering by selected bandpasses
    valid_mask = np.array([band in unique_bands for band in data['band']])
    
    # Check if we have any valid data points after filtering
    if not np.any(valid_mask):
        if selected_bandpasses:
            raise ValueError(f"No data points found for selected bandpasses {selected_bandpasses} in {sn_name}")
        else:
            raise ValueError(f"No valid data points found for {sn_name}")
    
    times = jnp.array(data['time'][valid_mask])
    fluxes = jnp.array(data['flux'][valid_mask])
    fluxerrs = jnp.array(data['fluxerr'][valid_mask])
    zps = jnp.array(data['zp'][valid_mask])
    band_indices = jnp.array([unique_bands.index(band) for band in data['band'][valid_mask]])
    
    # Print summary of selected data
    print(f"Using {len(unique_bands)} bandpasses: {unique_bands}")
    print(f"Total data points: {len(times)}")
    
    # Count data points for each band and remove bands with 0 points
    bands_to_keep = []
    bridges_to_keep = []
    for i, band in enumerate(unique_bands):
        band_count = np.sum(band_indices == i)
        print(f"  Band {band}: {band_count} points")
        
        # Keep the band if it has data points or was explicitly requested in the original selection
        if band_count > 0 or (original_selected_bandpasses is not None and band in original_selected_bandpasses):
            bands_to_keep.append(band)
            bridges_to_keep.append(bridges[i])
    
    # If we removed any bands, update the data structures
    if len(bands_to_keep) < len(unique_bands):
        print(f"Removing {len(unique_bands) - len(bands_to_keep)} bandpasses with no data points")
        
        # Create a mapping from old indices to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(
            [i for i, band in enumerate(unique_bands) if band in bands_to_keep]
        )}
        
        # Update band_indices
        new_band_indices = np.array([old_to_new[idx] for idx in band_indices])
        band_indices = jnp.array(new_band_indices)
        
        # Update unique_bands and bridges
        unique_bands = bands_to_keep
        bridges = tuple(bridges_to_keep)
        
        print(f"Final bandpasses: {unique_bands}")
    
    # Load redshift if requested
    fixed_z = None
    if fix_z:
        try:
            z, z_err, flag = load_redshift(sn_name)
            fixed_z = (z, z_err)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load redshift: {e}")
            fixed_z = None
    
    return times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z

# Load and process data
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = custom_load_and_process_data(
    sn_name, data_dir='hsf_DR1/', fix_z=fix_z, 
    selected_bandpasses=selected_bandpasses,
    custom_bandpass_files=custom_bandpass_files
)

# =============================================================================
# Set up parameter bounds and prior distributions for the standard (non‐anomaly)
# nested sampling version.
# =============================================================================
if fix_z:
    standard_param_bounds = {
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    if fit_sigma:
        standard_param_bounds['sigma'] = (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])
else:
    standard_param_bounds = {
        'z': (PRIOR_BOUNDS['z']['min'], PRIOR_BOUNDS['z']['max']),
        't0': (PRIOR_BOUNDS['t0']['min'], PRIOR_BOUNDS['t0']['max']),
        'x0': (PRIOR_BOUNDS['x0']['min'], PRIOR_BOUNDS['x0']['max']),
        'x1': (PRIOR_BOUNDS['x1']['min'], PRIOR_BOUNDS['x1']['max']),
        'c': (PRIOR_BOUNDS['c']['min'], PRIOR_BOUNDS['c']['max'])
    }
    if fit_sigma:
        standard_param_bounds['sigma'] = (PRIOR_BOUNDS['sigma']['min'], PRIOR_BOUNDS['sigma']['max'])

if fix_z:
    standard_prior_dists = {
        't0': distrax.Uniform(low=standard_param_bounds['t0'][0], high=standard_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=standard_param_bounds['x0'][0], high=standard_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=standard_param_bounds['x1'][0], high=standard_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=standard_param_bounds['c'][0], high=standard_param_bounds['c'][1])
    }
    if fit_sigma:
        standard_prior_dists['sigma'] = distrax.Uniform(low=standard_param_bounds['sigma'][0], high=standard_param_bounds['sigma'][1])
else:
    standard_prior_dists = {
        'z': distrax.Uniform(low=standard_param_bounds['z'][0], high=standard_param_bounds['z'][1]),
        't0': distrax.Uniform(low=standard_param_bounds['t0'][0], high=standard_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=standard_param_bounds['x0'][0], high=standard_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=standard_param_bounds['x1'][0], high=standard_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=standard_param_bounds['c'][0], high=standard_param_bounds['c'][1])
    }
    if fit_sigma:
        standard_prior_dists['sigma'] = distrax.Uniform(low=standard_param_bounds['sigma'][0], high=standard_param_bounds['sigma'][1])

# =============================================================================
# Set up parameter bounds and priors for the anomaly detection version.
# An extra parameter 'log_p' is included.
# =============================================================================
if fix_z:
    anomaly_param_bounds = {
        't0': (standard_param_bounds['t0'][0], standard_param_bounds['t0'][1]),
        'x0': (standard_param_bounds['x0'][0], standard_param_bounds['x0'][1]),
        'x1': (standard_param_bounds['x1'][0], standard_param_bounds['x1'][1]),
        'c': (standard_param_bounds['c'][0], standard_param_bounds['c'][1]),
        'log_p': (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    }
    if fit_sigma:
        anomaly_param_bounds['sigma'] = (standard_param_bounds['sigma'][0], standard_param_bounds['sigma'][1])
else:
    anomaly_param_bounds = {
        'z': (standard_param_bounds['z'][0], standard_param_bounds['z'][1]),
        't0': (standard_param_bounds['t0'][0], standard_param_bounds['t0'][1]),
        'x0': (standard_param_bounds['x0'][0], standard_param_bounds['x0'][1]),
        'x1': (standard_param_bounds['x1'][0], standard_param_bounds['x1'][1]),
        'c': (standard_param_bounds['c'][0], standard_param_bounds['c'][1]),
        'log_p': (PRIOR_BOUNDS['log_p']['min'], PRIOR_BOUNDS['log_p']['max'])
    }
    if fit_sigma:
        anomaly_param_bounds['sigma'] = (standard_param_bounds['sigma'][0], standard_param_bounds['sigma'][1])

if fix_z:
    anomaly_prior_dists = {
        't0': distrax.Uniform(low=anomaly_param_bounds['t0'][0], high=anomaly_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=anomaly_param_bounds['x0'][0], high=anomaly_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=anomaly_param_bounds['x1'][0], high=anomaly_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=anomaly_param_bounds['c'][0], high=anomaly_param_bounds['c'][1]),
        'log_p': distrax.Uniform(low=anomaly_param_bounds['log_p'][0], high=anomaly_param_bounds['log_p'][1])
    }
    if fit_sigma:
        anomaly_prior_dists['sigma'] = distrax.Uniform(low=anomaly_param_bounds['sigma'][0], high=anomaly_param_bounds['sigma'][1])
else:
    anomaly_prior_dists = {
        'z': distrax.Uniform(low=anomaly_param_bounds['z'][0], high=anomaly_param_bounds['z'][1]),
        't0': distrax.Uniform(low=anomaly_param_bounds['t0'][0], high=anomaly_param_bounds['t0'][1]),
        'x0': distrax.Uniform(low=anomaly_param_bounds['x0'][0], high=anomaly_param_bounds['x0'][1]),
        'x1': distrax.Uniform(low=anomaly_param_bounds['x1'][0], high=anomaly_param_bounds['x1'][1]),
        'c': distrax.Uniform(low=anomaly_param_bounds['c'][0], high=anomaly_param_bounds['c'][1]),
        'log_p': distrax.Uniform(low=anomaly_param_bounds['log_p'][0], high=anomaly_param_bounds['log_p'][1])
    }
    if fit_sigma:
        anomaly_prior_dists['sigma'] = distrax.Uniform(low=anomaly_param_bounds['sigma'][0], high=anomaly_param_bounds['sigma'][1])

# =============================================================================
# Standard likelihood functions (using salt3 multiband flux).
# =============================================================================
@jax.jit
def logprior_standard(params):
    """Calculate log prior probability for standard nested sampling."""
    params = jnp.atleast_2d(params)
    if fix_z:
        if fit_sigma:
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 2])
            logp_c  = standard_prior_dists['c'].log_prob(params[:, 3])
            logp_sigma = standard_prior_dists['sigma'].log_prob(params[:, 4])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma
        else:
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 2])
            logp_c  = standard_prior_dists['c'].log_prob(params[:, 3])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c
    else:
        if fit_sigma:
            logp_z  = standard_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 3])
            logp_c  = standard_prior_dists['c'].log_prob(params[:, 4])
            logp_sigma = standard_prior_dists['sigma'].log_prob(params[:, 5])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma
        else:
            logp_z  = standard_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = standard_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = standard_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = standard_prior_dists['x1'].log_prob(params[:, 3])
            logp_c  = standard_prior_dists['c'].log_prob(params[:, 4])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c
    return jnp.reshape(logp, (-1,))

@jax.jit
def logprior_anomaly(params):
    """Calculate log prior probability for anomaly detection nested sampling."""
    params = jnp.atleast_2d(params)
    if fix_z:
        if fit_sigma:
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 2])
            logp_c  = anomaly_prior_dists['c'].log_prob(params[:, 3])
            logp_sigma = anomaly_prior_dists['sigma'].log_prob(params[:, 4])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 5])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma + logp_logp
        else:
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 0])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 1])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 2])
            logp_c  = anomaly_prior_dists['c'].log_prob(params[:, 3])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 4])
            logp = logp_t0 + logp_x0 + logp_x1 + logp_c + logp_logp
    else:
        if fit_sigma:
            logp_z  = anomaly_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 3])
            logp_c  = anomaly_prior_dists['c'].log_prob(params[:, 4])
            logp_sigma = anomaly_prior_dists['sigma'].log_prob(params[:, 5])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 6])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_sigma + logp_logp
        else:
            logp_z  = anomaly_prior_dists['z'].log_prob(params[:, 0])
            logp_t0 = anomaly_prior_dists['t0'].log_prob(params[:, 1])
            logp_x0 = anomaly_prior_dists['x0'].log_prob(params[:, 2])
            logp_x1 = anomaly_prior_dists['x1'].log_prob(params[:, 3])
            logp_c  = anomaly_prior_dists['c'].log_prob(params[:, 4])
            logp_logp = anomaly_prior_dists['log_p'].log_prob(params[:, 5])
            logp = logp_z + logp_t0 + logp_x0 + logp_x1 + logp_c + logp_logp
    return jnp.reshape(logp, (-1,))

@jax.jit
def compute_single_loglikelihood_standard(params):
    """Compute Gaussian log likelihood for a single set of parameters (standard)."""
    if fix_z:
        if fit_sigma:
            t0, log_x0, x1, c, sigma = params
        else:
            t0, log_x0, x1, c = params
            sigma = 1.0  # Default value when not fitting sigma
        z = fixed_z[0]
    else:
        if fit_sigma:
            z, t0, log_x0, x1, c, sigma = params
        else:
            z, t0, log_x0, x1, c = params
            sigma = 1.0  # Default value when not fitting sigma
    x0 = 10 ** log_x0
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    eff_fluxerrs = sigma * fluxerrs
    chi2 = jnp.sum(((fluxes - model_fluxes) / eff_fluxerrs) ** 2)
    log_likelihood = -0.5 * (chi2 + jnp.sum(jnp.log(2 * jnp.pi * eff_fluxerrs ** 2)))
    return log_likelihood

@jax.jit
def compute_batch_loglikelihood_standard(params):
    params = jnp.atleast_2d(params)
    batch_loglike = jax.vmap(compute_single_loglikelihood_standard)(params)
    return jnp.reshape(batch_loglike, (-1,))

@jax.jit
def loglikelihood_standard(params):
    params = jnp.atleast_2d(params)
    batch_loglike = compute_batch_loglikelihood_standard(params)
    return batch_loglike

# =============================================================================
# Anomaly detection likelihood functions (using salt3 multiband flux).
# An extra parameter 'log_p' is used to weight the likelihood for anomalies.
# =============================================================================
@jax.jit
def compute_single_loglikelihood_anomaly(params):
    """Compute Gaussian log likelihood for a single set of parameters with anomaly detection."""
    if fix_z:
        if fit_sigma:
            t0, log_x0, x1, c, sigma, log_p = params
        else:
            t0, log_x0, x1, c, log_p = params
            sigma = 1.0  # Default value when not fitting sigma
        z = fixed_z[0]
    else:
        if fit_sigma:
            z, t0, log_x0, x1, c, sigma, log_p = params
        else:
            z, t0, log_x0, x1, c, log_p = params
            sigma = 1.0  # Default value when not fitting sigma
    x0 = 10 ** log_x0
    p = jnp.exp(log_p)  # Changed: Now using natural exponential
    param_dict = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, param_dict, zps=zps, zpsys='ab')
    model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]
    eff_fluxerrs = sigma * fluxerrs
    point_logL = -0.5 * (((fluxes - model_fluxes) / eff_fluxerrs) ** 2) - 0.5 * jnp.log(2 * jnp.pi * eff_fluxerrs ** 2) + jnp.log(1 - p)
    delta = jnp.max(jnp.abs(fluxes))  # Use maximum absolute flux value as delta
    emax = point_logL > (log_p - jnp.log(delta))  # Now consistent as both are natural logs
    logL = jnp.where(emax, point_logL, log_p - jnp.log(delta))
    total_logL = jnp.sum(logL)
    return total_logL, emax

@jax.jit
def compute_batch_loglikelihood_anomaly(params):
    params = jnp.atleast_2d(params)
    batch_loglike, batch_emax = jax.vmap(compute_single_loglikelihood_anomaly)(params)
    return jnp.reshape(batch_loglike, (-1,)), batch_emax

@jax.jit
def loglikelihood_anomaly(params):
    params = jnp.atleast_2d(params)
    batch_loglike, batch_emax = compute_batch_loglikelihood_anomaly(params)
    return batch_loglike

# =============================================================================
# Function to sample from the prior distributions.
# It chooses between the standard and anomaly priors based on the likelihood function.
# =============================================================================
def sample_from_priors(rng_key, n_samples, ll_fn=loglikelihood_standard):
    if ll_fn == loglikelihood_anomaly:
        if fix_z:
            if fit_sigma:
                keys = jax.random.split(rng_key, 6)
                return jnp.column_stack([
                    anomaly_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    anomaly_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    anomaly_prior_dists['sigma'].sample(seed=keys[4], sample_shape=(n_samples,)),
                    anomaly_prior_dists['log_p'].sample(seed=keys[5], sample_shape=(n_samples,))
                ])
            else:
                keys = jax.random.split(rng_key, 5)
                return jnp.column_stack([
                    anomaly_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    anomaly_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    anomaly_prior_dists['log_p'].sample(seed=keys[4], sample_shape=(n_samples,))
                ])
        else:  # Add case for anomaly model when fix_z is False
            if fit_sigma:
                keys = jax.random.split(rng_key, 7)
                return jnp.column_stack([
                    anomaly_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    anomaly_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    anomaly_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                    anomaly_prior_dists['sigma'].sample(seed=keys[5], sample_shape=(n_samples,)),
                    anomaly_prior_dists['log_p'].sample(seed=keys[6], sample_shape=(n_samples,))
                ])
            else:
                keys = jax.random.split(rng_key, 6)
                return jnp.column_stack([
                    anomaly_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    anomaly_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    anomaly_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    anomaly_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                    anomaly_prior_dists['log_p'].sample(seed=keys[5], sample_shape=(n_samples,))
                ])
    else:  # Standard case
        if fix_z:
            if fit_sigma:
                keys = jax.random.split(rng_key, 5)
                return jnp.column_stack([
                    standard_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    standard_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    standard_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    standard_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    standard_prior_dists['sigma'].sample(seed=keys[4], sample_shape=(n_samples,))
                ])
            else:
                keys = jax.random.split(rng_key, 4)
                return jnp.column_stack([
                    standard_prior_dists['t0'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    standard_prior_dists['x0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    standard_prior_dists['x1'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    standard_prior_dists['c'].sample(seed=keys[3], sample_shape=(n_samples,))
                ])
        else:
            if fit_sigma:
                keys = jax.random.split(rng_key, 6)
                return jnp.column_stack([
                    standard_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    standard_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    standard_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    standard_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    standard_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,)),
                    standard_prior_dists['sigma'].sample(seed=keys[5], sample_shape=(n_samples,))
                ])
            else:
                keys = jax.random.split(rng_key, 5)
                return jnp.column_stack([
                    standard_prior_dists['z'].sample(seed=keys[0], sample_shape=(n_samples,)),
                    standard_prior_dists['t0'].sample(seed=keys[1], sample_shape=(n_samples,)),
                    standard_prior_dists['x0'].sample(seed=keys[2], sample_shape=(n_samples,)),
                    standard_prior_dists['x1'].sample(seed=keys[3], sample_shape=(n_samples,)),
                    standard_prior_dists['c'].sample(seed=keys[4], sample_shape=(n_samples,))
                ])

# =============================================================================
# Set the total number of model parameters for the standard case.
# =============================================================================
n_params_total = 5 if fix_z else 6
num_mcmc_steps = n_params_total * NS_SETTINGS['num_mcmc_steps_multiplier']

# =============================================================================
# Function to run nested sampling.
# It initialises the BlackJAX nested sampler, runs the sampling loop,
# and saves output chains (and weighted anomaly indicators for the anomaly run).
# =============================================================================
def run_nested_sampling(ll_fn, output_prefix, sn_name, identifier="", num_iterations=NS_SETTINGS['max_iterations']):
    """Run nested sampling with output directories/files including supernova name.
    
    Args:
        ll_fn: Likelihood function to use
        output_prefix: Base prefix for output directory ('chains_standard' or 'chains_anomaly')
        sn_name: Name of the supernova (e.g. '20aai')
        identifier: Additional string to append to output directory and filenames
        num_iterations: Maximum number of iterations
    """
    # Create the main output directory
    output_dir = os.path.join("results", f"chains_{sn_name}{identifier}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running {output_prefix} nested sampling for {output_dir}...")
    algo = blackjax.ns.adaptive.nss(
        logprior_fn=logprior_anomaly if ll_fn == loglikelihood_anomaly else logprior_standard,
        loglikelihood_fn=ll_fn,
        n_delete=NS_SETTINGS['n_delete'],
        num_mcmc_steps=num_mcmc_steps,
    )
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = sample_from_priors(init_key, NS_SETTINGS['n_live'], ll_fn)
    print("Initial particles generated, shape: ", initial_particles.shape)
    if ll_fn == loglikelihood_standard:
        state = algo.init(initial_particles, compute_batch_loglikelihood_standard)
    else:
        state = algo.init(initial_particles, compute_batch_loglikelihood_anomaly)
    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point
    dead = []
    emax_values = []  # For anomaly detection runs
    for i in tqdm.trange(num_iterations):
        if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
            break
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        if ll_fn == loglikelihood_anomaly:
            # Handle multiple deleted points when n_delete > 1
            for j in range(len(dead_info.particles)):
                _, emax = compute_single_loglikelihood_anomaly(dead_info.particles[j])
                emax_values.append(emax)
        if i % 10 == 0:
            print(f"Iteration {i}: logZ = {state.sampler_state.logZ:.2f}")
    dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
    logw = log_weights(rng_key, dead)
    logZs = jax.scipy.special.logsumexp(logw, axis=0)
    print(f"Runtime evidence: {state.sampler_state.logZ:.2f}")
    print(f"Estimated evidence: {logZs.mean():.2f} +- {logZs.std():.2f}")
    if ll_fn == loglikelihood_standard:
        if fix_z:
            param_names = ['t0', 'log_x0', 'x1', 'c']
            if fit_sigma:
                param_names.append('sigma')
        else:
            param_names = ['z', 't0', 'log_x0', 'x1', 'c']
            if fit_sigma:
                param_names.append('sigma')
    else:
        if fix_z:
            param_names = ['t0', 'log_x0', 'x1', 'c']
            if fit_sigma:
                param_names.append('sigma')
            param_names.append('log_p')
        else:
            param_names = ['z', 't0', 'log_x0', 'x1', 'c']
            if fit_sigma:
                param_names.append('sigma')
            param_names.append('log_p')
    
    # Save chains with the correct filename
    chains_filename = f"{output_prefix}_dead-birth.txt"
    final_path = os.path.join(output_dir, chains_filename)
    
    # Extract data from dead info
    points = np.array(dead.particles)
    logls_death = np.array(dead.logL)
    logls_birth = np.array(dead.logL_birth)
    
    # Combine data: parameters, death likelihood, birth likelihood
    data = np.column_stack([points, logls_death, logls_birth])
    
    # Save directly to final location
    np.savetxt(final_path, data)
    print(f"Saved {data.shape[0]} samples to {final_path}")
    
    if ll_fn == loglikelihood_anomaly and emax_values:
        emax_array = jnp.stack(emax_values)
        print(f"emax_array shape: {emax_array.shape}")
        weights = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
        
        # Ensure weights and emax_array have compatible shapes
        if weights.ndim > 1:
            weights = weights[:, 0]
        
        # Check if shapes are compatible
        print(f"weights shape: {weights.shape}")
        
        if len(emax_array) != len(weights):
            print(f"Warning: emax_array length ({len(emax_array)}) doesn't match weights length ({len(weights)})")
            # Truncate or pad to make them compatible
            min_len = min(len(emax_array), len(weights))
            emax_array = emax_array[:min_len]
            weights = weights[:min_len]
            print(f"Adjusted to use first {min_len} elements")
        
        weighted_emax = jnp.zeros(emax_array.shape[1])
        for i in range(emax_array.shape[1]):
            weighted_emax = weighted_emax.at[i].set(jnp.sum(emax_array[:, i] * weights) / jnp.sum(weights))
        print(f"weighted_emax shape: {weighted_emax.shape}")
        emax_output_path = os.path.join(output_dir, f"{output_prefix}_weighted_emax.txt")
        np.savetxt(emax_output_path, weighted_emax)
        print(f"Saved weighted emax values to {emax_output_path}")

def get_n_params(ll_fn):
    """Get the number of parameters being fit."""
    if ll_fn == loglikelihood_standard:
        if fix_z:
            return 5 if fit_sigma else 4
        else:
            return 6 if fit_sigma else 5
    else:
        if fix_z:
            return 6 if fit_sigma else 5
        else:
            return 7 if fit_sigma else 6

def get_true_values(sn_name, data_dir='hsf_DR1/', selected_bandpasses=None):
    """
    Read the true values from the salt_fits.dat file for a given supernova.
    Only returns values for exactly matching bandpass combinations.
    
    Args:
        sn_name: Name of the supernova (e.g., '21yrf')
        data_dir: Base directory containing the data
        selected_bandpasses: List of bandpass names being used in the analysis
        
    Returns:
        Dictionary with true parameter values or None if no matching bandpass combination found
    """
    if selected_bandpasses is None:
        return None
        
    salt_fits_path = os.path.join(data_dir, 'Ia', sn_name, 'salt_fits.dat')
    
    try:
        # Read the entire file content
        with open(salt_fits_path, 'r') as f:
            lines = f.readlines()
            
        # Debug: Print raw file contents
        print("\nRaw file contents:")
        for line in lines[:5]:  # Print first 5 lines
            print(line.strip())
            
        # Parse header
        header = lines[0].strip().split()
        print("\nHeader:", header)
        
        # Sort bandpasses to ensure consistent ordering
        selected_bandpasses = sorted(selected_bandpasses)
        target_bps = '-'.join(selected_bandpasses)
        
        # Find matching row
        matching_row = None
        for line in lines[1:]:  # Skip header
            values = line.strip().split()
            if values[0] == target_bps:  # bps_used is always the first column
                matching_row = values
                break
                
        if matching_row is None:
            return None
            
        # Get the values from the correct columns
        # The columns are:
        # bps_used variant success reddening_law t0 e_t0 x0_mag e_x0_mag x1 e_x1 c e_c ...
        # So: t0 is column 4, x0_mag is column 6, x1 is column 8, c is column 10
        t0 = float(matching_row[3])  # t0 is in column 3 NOTE that the column headings are 
        x0_mag = float(matching_row[5])  # x0_mag is in column 5
        x1 = float(matching_row[8])  # x1 is in column 8
        c = float(matching_row[10])  # c is in column 10
        
        print("\nMatching row found:")
        print(f"bps_used: {matching_row[0]}")
        print(f"t0: {t0}")
        print(f"x0_mag: {x0_mag}")
        print(f"x1: {x1}")
        print(f"c: {c}")
        
        # Create dictionary with parameter values
        true_values = {
            't0': t0,
            'log_x0': -x0_mag / 2.5,  # Convert x0_mag to log_x0
            'x1': x1,
            'c': c
        }
            
        return true_values
        
    except Exception as e:
        print(f"\nError in get_true_values: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_x0mag_dm_relationship(output_dir):
    """
    Create a plot showing the relationship between x0_mag and DM.
    
    Args:
        output_dir: Directory to save the plot
    """
    try:
        # Create a range of x0_mag values
        x0_mag_values = np.linspace(8, 10, 100)
        
        # Calculate corresponding DM values (DM = x0_mag + 21.01)
        dm_values = x0_mag_values + 21.01
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(x0_mag_values, dm_values, 'b-', linewidth=2)
        plt.xlabel('x0_mag = -2.5 * log10(x0)', fontsize=12)
        plt.ylabel('Distance Modulus (DM)', fontsize=12)
        plt.title('Relationship between x0_mag and Distance Modulus', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add text explaining the relationship
        plt.text(0.05, 0.95, 'DM = x0_mag + 21.01', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'x0mag_dm_relationship.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved x0_mag vs DM relationship plot to {os.path.join(output_dir, 'x0mag_dm_relationship.png')}")
        
    except Exception as e:
        print(f"Warning: Failed to create x0_mag vs DM relationship plot - {str(e)}")
        plt.close()

def plot_samples_x0mag_dm(standard_samples, anomaly_samples, true_values, output_dir):
    """
    Create a scatter plot of the samples showing x0_mag vs DM.
    
    Args:
        standard_samples: Samples from standard nested sampling
        anomaly_samples: Samples from anomaly nested sampling
        true_values: Dictionary with true parameter values
        output_dir: Directory to save the plot
    """
    try:
        if (standard_samples is None or 'log_x0' not in standard_samples.columns) and \
           (anomaly_samples is None or 'log_x0' not in anomaly_samples.columns):
            print("Warning: log_x0 not found in samples - skipping x0_mag vs DM scatter plot")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Calculate x0_mag and DM for standard samples
        if standard_samples is not None and 'log_x0' in standard_samples.columns:
            x0_mag_std = -2.5 * standard_samples['log_x0']
            dm_std = x0_mag_std + 21.01
            plt.scatter(x0_mag_std, dm_std, alpha=0.5, label='Standard', s=10)
        
        # Calculate x0_mag and DM for anomaly samples
        if anomaly_samples is not None and 'log_x0' in anomaly_samples.columns:
            x0_mag_anom = -2.5 * anomaly_samples['log_x0']
            dm_anom = x0_mag_anom + 21.01
            plt.scatter(x0_mag_anom, dm_anom, alpha=0.5, label='Anomaly', s=10)
        
        # Add true value if available
        if true_values and 'log_x0' in true_values:
            true_x0_mag = -2.5 * true_values['log_x0']
            true_dm = true_x0_mag + 21.01
            plt.scatter([true_x0_mag], [true_dm], color='red', marker='*', s=200, 
                       label='True Value', zorder=10)
        
        # Add the DM = x0_mag + 21.01 line
        x_range = plt.xlim()
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        plt.plot(x_vals, x_vals + 21.01, 'k--', label='DM = x0_mag + 21.01')
        
        plt.xlabel('x0_mag = -2.5 * log10(x0)', fontsize=12)
        plt.ylabel('Distance Modulus (DM = x0_mag + 21.01)', fontsize=12)
        plt.title('Relationship between x0_mag and Distance Modulus in Samples', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'samples_x0mag_dm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved samples x0_mag vs DM plot to {os.path.join(output_dir, 'samples_x0mag_dm.png')}")
        
    except Exception as e:
        print(f"Warning: Failed to create samples x0_mag vs DM plot - {str(e)}")
        plt.close()

if __name__ == "__main__":
    # Add an identifier for this run (e.g. date, version, etc)
    identifier = "_tes"  # You can modify this or pass it as a command line argument
    
    print("\nRunning anomaly detection version...")
    n_params = get_n_params(loglikelihood_anomaly)
    num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']
    anomaly_samples = run_nested_sampling(loglikelihood_anomaly, "chains_anomaly", sn_name, identifier)

    print("Running standard version...")
    n_params = get_n_params(loglikelihood_standard)
    num_mcmc_steps = n_params * NS_SETTINGS['num_mcmc_steps_multiplier']
    standard_samples = run_nested_sampling(loglikelihood_standard, "chains_standard", sn_name, identifier)

    print("\nGenerating plots...")
    # Define output directory
    output_dir = f'results/chains_{sn_name}{identifier}'
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define parameter names based on settings
    if fix_z:
        base_params = ['t0', 'log_x0', 'x1', 'c']
    else:
        base_params = ['z', 't0', 'log_x0', 'x1', 'c']

    if fit_sigma:
        base_params.append('sigma')
        
    # Get true values from salt_fits.dat with matching bandpasses
    true_values = get_true_values(sn_name, selected_bandpasses=selected_bandpasses)
    print(f"True values from salt_fits.dat: {true_values}")

    # Try to load weighted emax values and create initial plot
    try:
        weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
        
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(weighted_emax)), weighted_emax, 'k-', linewidth=2)
        plt.fill_between(np.arange(len(weighted_emax)), 0, weighted_emax, alpha=0.3)
        plt.xlabel('Data Point Number')
        plt.ylabel('Weighted Emax')
        plt.title('Weighted Emax by Data Point')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/weighted_emax.png', dpi=300, bbox_inches='tight')
    except FileNotFoundError:
        print("Warning: Weighted emax file not found - skipping initial emax plot")

    # Try to load standard chains
    try:
        standard_samples = read_chains(f'{output_dir}/chains_standard', columns=base_params)
        have_standard = True
    except FileNotFoundError:
        print("Warning: Standard chains not found - some plots will be incomplete")
        have_standard = False
        standard_samples = None

    # Try to load anomaly chains with log_p parameter
    try:
        anomaly_params = base_params + ['log_p']
        anomaly_samples = read_chains(f'{output_dir}/chains_anomaly', columns=anomaly_params)
        have_anomaly = True
    except FileNotFoundError:
        print("Warning: Anomaly chains not found - some plots will be incomplete")
        have_anomaly = False
        anomaly_samples = None

    # Use the appropriate parameter names for plotting
    param_names = base_params  # Only plot the common parameters between both chains

    # Only create corner plot if we have at least one set of chains
    if have_standard or have_anomaly:
        # Create overlaid corner plot
        try:
            # Convert log_x0 to x0_mag in the samples for plotting
            plot_param_names = param_names.copy()
            
            # Replace log_x0 with x0_mag in the parameter names list
            if 'log_x0' in plot_param_names:
                plot_param_names[plot_param_names.index('log_x0')] = 'x0_mag'
            
            # Create a copy of the samples with x0_mag instead of log_x0
            if have_standard:
                standard_plot_samples = standard_samples.copy()
                if 'log_x0' in standard_plot_samples.columns:
                    standard_plot_samples['x0_mag'] = -2.5 * standard_plot_samples['log_x0']
                    standard_plot_samples = standard_plot_samples.drop(columns=['log_x0'])
            
            if have_anomaly:
                anomaly_plot_samples = anomaly_samples.copy()
                if 'log_x0' in anomaly_plot_samples.columns:
                    anomaly_plot_samples['x0_mag'] = -2.5 * anomaly_plot_samples['log_x0']
                    anomaly_plot_samples = anomaly_plot_samples.drop(columns=['log_x0'])
            
            # Convert true values from log_x0 to x0_mag if needed
            plot_true_values = true_values.copy() if true_values else {}
            if 'log_x0' in plot_true_values:
                plot_true_values['x0_mag'] = -2.5 * plot_true_values['log_x0']
                del plot_true_values['log_x0']
            
            fig, axes = make_2d_axes(plot_param_names, figsize=(10, 10), facecolor='w')
            
            # Plot standard samples if available
            if have_standard:
                try:
                    standard_plot_samples.plot_2d(axes, alpha=0.7, label="Standard")
                except Exception as e:
                    print(f"Warning: Failed to plot standard samples in corner plot - {str(e)}")
            
            # Plot anomaly samples if available
            if have_anomaly:
                try:
                    anomaly_plot_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
                except Exception as e:
                    print(f"Warning: Failed to plot anomaly samples in corner plot - {str(e)}")
            
            # Add true values as reference lines if available
            if plot_true_values:
                for param in plot_param_names:
                    if param in plot_true_values:
                        # Add individual parameter lines
                        axes.axlines({param: plot_true_values[param]}, c='green', linestyle='--', 
                                    linewidth=2, alpha=1.0, zorder=10, label='True values')
                
                # Add joint parameter lines for each pair of parameters
                for i, param1 in enumerate(plot_param_names):
                    if param1 not in plot_true_values:
                        continue
                    for param2 in plot_param_names[i+1:]:
                        if param2 not in plot_true_values:
                            continue
                        axes.axlines({param1: plot_true_values[param1], param2: plot_true_values[param2]}, 
                                    c='green', linestyle='--', linewidth=2, alpha=1.0, zorder=10)
            
            # Add legend with unique labels
            handles, labels = axes.iloc[-1, 0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes.iloc[-1, 0].legend(by_label.values(), by_label.keys(), 
                                   bbox_to_anchor=(len(axes)/2, len(axes)), 
                                   loc='lower center', ncol=3)
            
            plt.savefig(f'{output_dir}/corner_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to create corner comparison plot - {str(e)}")

    # Create anomaly corner plot only if we have anomaly chains
    if have_anomaly and 'log_p' in anomaly_samples.columns:
        try:
            # Create parameter list with x0_mag instead of log_x0
            log_p_params = base_params.copy()
            if 'log_x0' in log_p_params:
                log_p_params[log_p_params.index('log_x0')] = 'x0_mag'
            log_p_params.append('log_p')
            
            # Use the previously created anomaly_plot_samples that has x0_mag
            fig, axes = make_2d_axes(log_p_params, figsize=(12, 12), facecolor='w')
            anomaly_plot_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
            
            # Add true values as reference lines if available
            if plot_true_values:
                for param in log_p_params:
                    if param in plot_true_values:
                        # Add individual parameter lines
                        axes.axlines({param: plot_true_values[param]}, c='green', linestyle='--', 
                                    linewidth=2, alpha=1.0, zorder=10, label='True values')
                
                # Add joint parameter lines for each pair of parameters
                for i, param1 in enumerate(log_p_params):
                    if param1 not in plot_true_values:
                        continue
                    for param2 in log_p_params[i+1:]:
                        if param2 not in plot_true_values:
                            continue
                        axes.axlines({param1: plot_true_values[param1], param2: plot_true_values[param2]}, 
                                    c='green', linestyle='--', linewidth=2, alpha=1.0, zorder=10)
            
            plt.suptitle('Anomaly Detection Corner Plot (including log_p)', fontsize=14)
            
            # Add legend with unique labels
            handles, labels = axes.iloc[-1, 0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes.iloc[-1, 0].legend(by_label.values(), by_label.keys(), 
                                   bbox_to_anchor=(len(axes)/2, len(axes)), 
                                   loc='lower center', ncol=2)
            
            plt.savefig(f'{output_dir}/corner_anomaly_logp.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to create anomaly corner plot - {str(e)}")

    def get_model_curve(samples, percentile=50):
        """Get model curve for given percentile of parameters."""
        params = {}
        for param in param_names:
            if param != 'log_p':  # Skip logp as it's not needed for the model
                params[param] = float(np.percentile(samples[param], percentile))
        if 'log_x0' in params:
            # Convert log_x0 to x0 for the model calculation
            params['x0'] = 10**params['log_x0']
            # Also store x0_mag for plotting
            params['x0_mag'] = -2.5 * params['log_x0']
            del params['log_x0']  # Remove log_x0 as we now have x0
        if fix_z:
            params['z'] = fixed_z[0]
        if not fit_sigma:
            params['sigma'] = 1.0  # Add default sigma if not fitted
        return params

    try:
        # Create time grid for smooth model curves
        t_min = np.min(times) - 5
        t_max = np.max(times) + 5
        t_grid = np.linspace(t_min, t_max, 100)

        # Get unique bands
        unique_bands = np.unique(band_indices)
        n_bands = len(unique_bands)

        # Set up the plot with two subplots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.1)

        # Main light curve plot
        ax1 = plt.subplot(gs[0])

        # Define colours for each band
        default_colours = ['g', 'c', 'orange', 'r']
        default_markers = ['o', 's', 'D', '^']
        
        # Ensure we have enough colours and markers for all bands
        if n_bands > len(default_colours):
            colours = plt.cm.tab10(np.linspace(0, 1, n_bands))
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '8'][:n_bands]
        else:
            colours = default_colours[:n_bands]
            markers = default_markers[:n_bands]

        # Load weighted emax values first to identify anomalous points
        try:
            weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
            plotting_threshold = 0.2
            
            # Count total anomalous points
            total_anomalous_points = 0
            
            # Create time points that match the actual data points
            all_times = np.sort(np.unique(times))
            if len(all_times) != len(weighted_emax):
                print(f"Warning: Number of unique time points ({len(all_times)}) "
                      f"doesn't match number of weighted_emax values ({len(weighted_emax)})")
            
            # Create time points for the emax plot - using actual data time points
            emax_times = all_times
        except FileNotFoundError:
            print("Warning: Weighted emax file not found")
            weighted_emax = None

        # Plot data points for each band
        try:
            for i, band_idx in enumerate(unique_bands):
                mask = band_indices == band_idx
                band_times = times[mask]
                band_fluxes = fluxes[mask]
                band_errors = fluxerrs[mask]

                if weighted_emax is not None:
                    # Map each data point to its index in the sorted unique times
                    time_indices = np.searchsorted(all_times, band_times)
                    # Ensure indices are within bounds
                    time_indices = np.clip(time_indices, 0, len(weighted_emax) - 1)
                    # Get the emax value for each point
                    point_emax = weighted_emax[time_indices]
                    
                    # Determine which points are anomalous
                    normal_mask = point_emax >= plotting_threshold
                    anomaly_mask = point_emax < plotting_threshold
                    
                    # Update total count
                    total_anomalous_points += np.sum(anomaly_mask)

                    # Plot normal points
                    if np.any(normal_mask):
                        ax1.errorbar(band_times[normal_mask], band_fluxes[normal_mask], 
                                   yerr=band_errors[normal_mask],
                                   fmt=markers[i], color=colours[i], 
                                   label=f'Band {i} Data',
                                   markersize=8, alpha=0.6)
                    
                    # Plot anomalous points with star markers
                    if np.any(anomaly_mask):
                        label = f'Band {i} Anomalous' if np.any(normal_mask) else f'Band {i} Data'
                        ax1.errorbar(band_times[anomaly_mask], band_fluxes[anomaly_mask], 
                                   yerr=band_errors[anomaly_mask],
                                   fmt='*', color=colours[i], 
                                   label=label,
                                   markersize=15, alpha=0.8)
                else:
                    # Plot all points normally if no weighted_emax available
                    ax1.errorbar(band_times, band_fluxes, yerr=band_errors,
                               fmt=markers[i], color=colours[i], label=f'Band {i} Data',
                               markersize=8, alpha=0.6)
        except Exception as e:
            print(f"Warning: Failed to plot some data points - {str(e)}")

        # Calculate and plot model curves for both standard and anomaly if available
        if have_standard or have_anomaly:
            for name, samples, has_samples in [
                ("Standard", standard_samples, have_standard), 
                ("Anomaly", anomaly_samples, have_anomaly)
            ]:
                if has_samples:
                    try:
                        params = get_model_curve(samples)
                        linestyle = '--' if name == "Standard" else '-'
                        
                        for i in range(n_bands):
                            try:
                                # Calculate model fluxes
                                model_fluxes = optimized_salt3_multiband_flux(
                                    jnp.array(t_grid),
                                    bridges,
                                    params,
                                    zps=zps,
                                    zpsys='ab'
                                )
                                
                                # Extract fluxes for this band
                                band_fluxes = model_fluxes[:, i]
                                
                                # Plot model curve
                                ax1.plot(t_grid, band_fluxes, linestyle, color=colours[i], 
                                        label=f'Band {i} {name}', linewidth=2, alpha=0.8)
                            except Exception as e:
                                print(f"Warning: Failed to plot {name} model curve for band {i} - {str(e)}")
                                continue
                    except Exception as e:
                        print(f"Warning: Failed to plot {name} model curves - {str(e)}")
                        continue

        # Add labels and title to main plot
        ax1.set_xlabel('MJD', fontsize=12)
        ax1.set_ylabel('Flux', fontsize=12)
        title = 'Light Curve Fit Comparison'
        if fix_z:
            title += f' (z = {fixed_z[0]:.4f})'
        ax1.set_title(title, fontsize=14)

        # Add legend to main plot
        ax1.legend(ncol=2, fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Try to load weighted emax values and create subplot
        try:
            weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
            ax2 = plt.subplot(gs[1])
            
            # Use actual data time points for the emax plot
            ax2.plot(all_times, weighted_emax, 'k-', linewidth=2)
            ax2.fill_between(all_times, 0, weighted_emax, alpha=0.3, color='gray')
            ax2.set_xlabel('MJD', fontsize=12)
            ax2.set_ylabel('Emax', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Add horizontal line at threshold
            ax2.axhline(y=plotting_threshold, color='r', linestyle='--', alpha=0.5, 
                       label=f'Plotting threshold ({plotting_threshold}) - {total_anomalous_points} points below')
            ax2.legend()

            # Ensure x-axis limits match between plots
            xlim = ax1.get_xlim()
            ax2.set_xlim(xlim)

            # Remove x-axis labels from top plot
            ax1.set_xlabel('')
        except FileNotFoundError:
            print("Warning: Weighted emax file not found - skipping emax subplot")
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Failed to create emax subplot - {str(e)}")
            plt.tight_layout()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{output_dir}/light_curve_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Warning: Failed to create light curve comparison plot - {str(e)}")
        plt.close('all')

    # Save parameter statistics to a text file if chains are available
    if have_standard or have_anomaly:
        stats_text = ["Parameter Statistics Comparison:", "-" * 50]
        
        # Add statistics for all parameters
        for param in param_names:
            if have_standard:
                std_mean = standard_samples[param].mean()
                std_std = standard_samples[param].std()
                stats_text.append(f"\n{param}:")
                stats_text.append(f"  Standard: {std_mean:.6f} ± {std_std:.6f}")
            if have_anomaly:
                anom_mean = anomaly_samples[param].mean()
                anom_std = anomaly_samples[param].std()
                stats_text.append(f"  Anomaly:  {anom_mean:.6f} ± {anom_std:.6f}")
        
        # Add x0_mag statistics
        if 'log_x0' in param_names:
            stats_text.append(f"\nx0_mag (calculated from log_x0):")
            if have_standard:
                x0_mag_std = -2.5 * standard_samples['log_x0']
                stats_text.append(f"  Standard: {x0_mag_std.mean():.6f} ± {x0_mag_std.std():.6f}")
            if have_anomaly:
                x0_mag_anom = -2.5 * anomaly_samples['log_x0']
                stats_text.append(f"  Anomaly:  {x0_mag_anom.mean():.6f} ± {x0_mag_anom.std():.6f}")
            
            # Add DM estimate (x0_mag + 21.01)
            stats_text.append(f"\nDistance Modulus (DM = x0_mag + 21.01):")
            if have_standard:
                dm_std = x0_mag_std + 21.01
                stats_text.append(f"  Standard: {dm_std.mean():.6f} ± {dm_std.std():.6f}")
            if have_anomaly:
                dm_anom = x0_mag_anom + 21.01
                stats_text.append(f"  Anomaly:  {dm_anom.mean():.6f} ± {dm_anom.std():.6f}")

        # Add log_p statistics for anomaly case if available
        if have_anomaly and 'log_p' in anomaly_samples.columns:
            stats_text.extend([
                "\nlog_p (Anomaly only):",
                f"  Mean: {anomaly_samples['log_p'].mean():.6f} ± {anomaly_samples['log_p'].std():.6f}",
                f"  Max: {anomaly_samples['log_p'].max():.6f}",
                f"  Min: {anomaly_samples['log_p'].min():.6f}"
            ])

        # Save statistics
        stats_text = '\n'.join(stats_text)
        with open(f'{output_dir}/parameter_statistics.txt', 'w') as f:
            f.write(stats_text)

    # Create x0_mag vs DM relationship plot
    plot_x0mag_dm_relationship(output_dir)
    
    # Create scatter plot of the samples showing x0_mag vs DM
    plot_samples_x0mag_dm(standard_samples, anomaly_samples, true_values, output_dir)
    