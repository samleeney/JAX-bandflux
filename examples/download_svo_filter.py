#!/usr/bin/env python
"""
Comprehensive filter profile management for JAX-bandflux.

This script provides utilities for:
1. Downloading filter profiles from the SVO Filter Profile Service
2. Creating synthetic filter profiles when needed
3. Registering and using custom bandpasses in JAX-bandflux
4. Demonstrating how to use custom bandpasses in SALT3 model fits

Usage:
    python download_svo_filter.py [--filter FILTER_ID] [--output OUTPUT_DIR] [--force]
    python download_svo_filter.py --list
    python download_svo_filter.py --example [--sn SN_NAME] [--bandpass-name NAME]
    python download_svo_filter.py --synthetic [--output OUTPUT] [--points POINTS]

Options:
    --filter FILTER_ID    SVO filter identifier (e.g., 'UKIRT/WFCAM.J'). Default: UKIRT/WFCAM.J
    --output OUTPUT_DIR   Directory to save the filter profile to. Default: filter_data
    --force               Force download even if the file already exists
    --list                List some common filter IDs available in the SVO database
    --example             Run an example of using a custom bandpass in a SALT3 model fit
    --sn SN_NAME          Supernova name to load data for in the example. Default: 19vnk
    --bandpass-name NAME  Name to register the custom bandpass under. Default: custom_J
    --synthetic           Create a synthetic filter profile instead of downloading from SVO
    --points POINTS       Number of points in the synthetic filter profile. Default: 100
"""

import os
import sys
import argparse
import numpy as np
import requests
from io import StringIO
from pathlib import Path

# Constants for synthetic WFCAM J filter
WFCAM_J_WAVE_MIN = 11700  # Angstroms
WFCAM_J_WAVE_MAX = 13300  # Angstroms

def download_svo_filter(filter_id, output_dir=None, force_download=False):
    """
    Download a filter profile from the SVO Filter Profile Service.
    
    Parameters
    ----------
    filter_id : str
        The SVO filter identifier, e.g., 'UKIRT/WFCAM.J'
    output_dir : str, optional
        Directory to save the downloaded filter file. If None, the file is not saved.
    force_download : bool, optional
        If True, download the filter even if it already exists locally.
        
    Returns
    -------
    tuple
        A tuple containing:
        - wave: numpy array of wavelengths in Angstroms
        - trans: numpy array of transmission values (normalized to peak of 1.0)
        - success: boolean indicating whether the download was successful
    """
    # Construct the URL for the filter data
    base_url = "http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id="
    url = base_url + filter_id
    
    # Check if the file already exists locally
    local_filename = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        local_filename = os.path.join(output_dir, f"{filter_id.replace('/', '_')}.dat")
        if os.path.exists(local_filename) and not force_download:
            print(f"Loading filter {filter_id} from local file {local_filename}")
            try:
                data = np.loadtxt(local_filename)
                wave, trans = data[:, 0], data[:, 1]
                return wave, trans, True
            except Exception as e:
                print(f"Error loading local file: {e}")
                # Continue with download
    
    # Download the filter data
    print(f"Downloading filter {filter_id} from SVO")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the data
        data_str = response.text
        
        # Check if the response contains actual data or an error message
        if "No filter" in data_str or "Error" in data_str:
            print(f"Filter {filter_id} not found in SVO database")
            return None, None, False
        
        # SVO data format has a header with '#' comments, so we need to skip those lines
        data_io = StringIO(data_str)
        lines = data_io.readlines()
        data_lines = []
        for line in lines:
            if not line.startswith('#') and line.strip():
                data_lines.append(line)
        
        # Check if we have any data lines
        if not data_lines:
            print(f"No data found for filter {filter_id}")
            return None, None, False
        
        # Convert to numpy arrays
        data_str = ''.join(data_lines)
        data_io = StringIO(data_str)
        data = np.loadtxt(data_io)
        
        # Extract wavelength (in Angstroms) and transmission
        wave, trans = data[:, 0], data[:, 1]
        
        # Normalize transmission to peak of 1.0
        trans = trans / np.max(trans)
        
        # Save the data locally if output_dir is provided
        if local_filename is not None:
            print(f"Saving filter {filter_id} to {local_filename}")
            np.savetxt(local_filename, np.column_stack((wave, trans)), 
                      header=f"Wavelength (Angstroms)\tTransmission\nFilter: {filter_id}\nDownloaded from SVO Filter Profile Service")
        
        return wave, trans, True
    
    except requests.RequestException as e:
        print(f"Error downloading filter {filter_id}: {e}")
        return None, None, False

def create_synthetic_filter(wave_min, wave_max, num_points=100, filter_name="synthetic"):
    """
    Create a synthetic filter profile with a Gaussian-like transmission curve.
    
    Parameters
    ----------
    wave_min : float
        Minimum wavelength in Angstroms
    wave_max : float
        Maximum wavelength in Angstroms
    num_points : int, optional
        Number of points in the filter profile. Default is 100.
    filter_name : str, optional
        Name of the filter for display purposes. Default is "synthetic".
        
    Returns
    -------
    tuple
        A tuple containing:
        - wave: numpy array of wavelengths in Angstroms
        - trans: numpy array of transmission values (normalized to peak of 1.0)
    """
    # Create a simple transmission curve with specified number of points
    wave = np.linspace(wave_min, wave_max, num_points)
    
    # Create a simplified transmission curve (approximately Gaussian)
    center = (wave_min + wave_max) / 2
    width = (wave_max - wave_min) / 4
    trans = np.exp(-((wave - center) ** 2) / (2 * width ** 2))
    
    # Normalize to peak of 1.0
    trans = trans / np.max(trans)
    
    print(f"Created synthetic {filter_name} filter profile")
    print(f"Wavelength range: {wave_min} - {wave_max} Angstroms")
    print(f"Number of points: {num_points}")
    
    return wave, trans

def save_filter_data(wave, trans, output_file, filter_name="filter"):
    """
    Save filter data to a file.
    
    Parameters
    ----------
    wave : array-like
        Wavelength values in Angstroms
    trans : array-like
        Transmission values (normalized to peak of 1.0)
    output_file : str
        Path to the output file
    filter_name : str, optional
        Name of the filter for the header. Default is "filter".
    """
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the data
    np.savetxt(output_file, np.column_stack((wave, trans)), 
              header=f"Wavelength (Angstroms)\tTransmission\nFilter: {filter_name}")
    
    print(f"Saved filter data to {output_file}")

def list_common_filters():
    """Print a list of common filter IDs available in the SVO database."""
    common_filters = [
        # UKIRT WFCAM filters
        {"id": "UKIRT/WFCAM.J", "description": "UKIRT WFCAM J-band filter (1.17-1.33 μm)"},
        {"id": "UKIRT/WFCAM.H", "description": "UKIRT WFCAM H-band filter (1.49-1.78 μm)"},
        {"id": "UKIRT/WFCAM.K", "description": "UKIRT WFCAM K-band filter (2.03-2.37 μm)"},
        
        # SDSS filters
        {"id": "SLOAN/SDSS.u", "description": "SDSS u-band filter (3000-4000 Å)"},
        {"id": "SLOAN/SDSS.g", "description": "SDSS g-band filter (4000-5500 Å)"},
        {"id": "SLOAN/SDSS.r", "description": "SDSS r-band filter (5500-7000 Å)"},
        {"id": "SLOAN/SDSS.i", "description": "SDSS i-band filter (7000-8500 Å)"},
        {"id": "SLOAN/SDSS.z", "description": "SDSS z-band filter (8500-10000 Å)"},
        
        # ZTF filters
        {"id": "ZTF/ZTF.g", "description": "ZTF g-band filter (4000-5500 Å)"},
        {"id": "ZTF/ZTF.r", "description": "ZTF r-band filter (5500-7000 Å)"},
        {"id": "ZTF/ZTF.i", "description": "ZTF i-band filter (7000-8500 Å)"},
        
        # 2MASS filters
        {"id": "2MASS/2MASS.J", "description": "2MASS J-band filter (1.1-1.4 μm)"},
        {"id": "2MASS/2MASS.H", "description": "2MASS H-band filter (1.5-1.8 μm)"},
        {"id": "2MASS/2MASS.Ks", "description": "2MASS Ks-band filter (2.0-2.3 μm)"},
    ]
    
    print("\nCommon filter IDs available in the SVO database:")
    print("================================================")
    for filter_info in common_filters:
        print(f"{filter_info['id']:<20} - {filter_info['description']}")
    
    print("\nFor a complete list of available filters, visit:")
    print("http://svo2.cab.inta-csic.es/theory/fps/")

def run_custom_bandpass_example(filter_id, bandpass_name, sn_name, output_dir):
    """
    Run an example of using a custom bandpass in a SALT3 model fit.
    
    Parameters
    ----------
    filter_id : str
        The SVO filter identifier, e.g., 'UKIRT/WFCAM.J'
    bandpass_name : str
        Name to register the custom bandpass under
    sn_name : str
        Name of the supernova to load data for
    output_dir : str
        Directory to save the filter profile to
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax_supernovae.bandpasses import Bandpass, register_bandpass, register_all_bandpasses
        from jax_supernovae.salt3 import precompute_bandflux_bridge, salt3_bandflux
        from jax_supernovae.data import load_hsf_data
    except ImportError as e:
        print(f"Error: {e}")
        print("This example requires JAX and JAX-bandflux to be installed.")
        return 1
    
    print(f"Loading data for {sn_name} with custom bandpass {bandpass_name} from {filter_id}")
    
    # Download the filter profile
    wave, trans, success = download_svo_filter(filter_id, output_dir)
    if not success:
        print(f"Failed to download {filter_id} filter profile")
        return 1
    
    # Load data
    try:
        data = load_hsf_data(sn_name, base_dir="hsf_DR1")
    except Exception as e:
        print(f"Error loading data for {sn_name}: {e}")
        print("Make sure the data directory exists and contains the required files.")
        return 1
    
    # Register standard bandpasses
    bandpass_dict, bridges_dict = register_all_bandpasses()
    
    # Create and register the custom bandpass
    try:
        custom_bandpass = Bandpass(wave=jnp.array(wave), trans=jnp.array(trans))
        register_bandpass(bandpass_name, custom_bandpass, force=True)
        bandpass_dict[bandpass_name] = custom_bandpass
        bridges_dict[bandpass_name] = precompute_bandflux_bridge(custom_bandpass)
        print(f"Successfully registered custom bandpass '{bandpass_name}' from {filter_id}")
    except Exception as e:
        print(f"Warning: Failed to create custom bandpass: {e}")
    
    # Get unique bands and their bridges
    unique_bands = []
    bridges = []
    for band in np.unique(data['band']):
        if band in bridges_dict:
            unique_bands.append(band)
            bridges.append(bridges_dict[band])
    
    # Add the custom bandpass if it's not in the data
    if bandpass_name not in unique_bands and bandpass_name in bridges_dict:
        unique_bands.append(bandpass_name)
        bridges.append(bridges_dict[bandpass_name])
        print(f"Added {bandpass_name} band to the list of unique bands")
    
    # Convert bridges to tuple for JIT compatibility
    bridges = tuple(bridges)

    # Set up data arrays
    valid_mask = np.array([band in unique_bands for band in data['band']])
    
    # Check if we have any valid data points
    if not np.any(valid_mask):
        print(f"No valid data points found for {sn_name}")
        return 1
    
    times = jnp.array(data['time'][valid_mask])
    fluxes = jnp.array(data['flux'][valid_mask])
    fluxerrs = jnp.array(data['fluxerr'][valid_mask])
    zps = jnp.array(data['zp'][valid_mask])
    band_indices = jnp.array([unique_bands.index(band) for band in data['band'][valid_mask]])
    
    # Print summary of selected data
    print(f"Using {len(unique_bands)} bandpasses: {unique_bands}")
    print(f"Total data points: {len(times)}")
    for i, band in enumerate(unique_bands):
        band_count = np.sum(band_indices == i)
        print(f"  Band {band}: {band_count} points")
    
    print("\nData loaded successfully with custom bandpass.")
    print("\nExample of how to use the custom bandpass in a SALT3 model fit:")
    print("```python")
    print("# Define model parameters")
    print("params = {")
    print("    'z': 0.05,  # redshift")
    print("    't0': 59000.0,  # time of maximum light")
    print("    'x0': -3.0,  # amplitude parameter")
    print("    'x1': 0.0,  # shape parameter")
    print("    'c': 0.0  # color parameter")
    print("}")
    print("")
    print("# Compute model fluxes for all observations")
    print("model_fluxes = []")
    print("for i, (t, band_idx, zp) in enumerate(zip(times, band_indices, zps)):")
    print("    flux = salt3_bandflux(t, bridges[band_idx], params, zp=zp)")
    print("    model_fluxes.append(flux)")
    print("")
    print("# Convert to a JAX array and calculate the chi-squared statistic")
    print("model_fluxes = jnp.array(model_fluxes)")
    print("chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)")
    print("```")
    
    return 0

def main():
    """Main function to handle various filter profile operations."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Comprehensive filter profile management for JAX-bandflux.")
    parser.add_argument("--filter", default="UKIRT/WFCAM.J", help="SVO filter identifier (e.g., 'UKIRT/WFCAM.J')")
    parser.add_argument("--output", default="filter_data", help="Directory to save the filter profile to")
    parser.add_argument("--force", action="store_true", help="Force download even if the file already exists")
    parser.add_argument("--list", action="store_true", help="List some common filter IDs available in the SVO database")
    parser.add_argument("--example", action="store_true", help="Run an example of using a custom bandpass in a SALT3 model fit")
    parser.add_argument("--sn", default="19vnk", help="Supernova name to load data for in the example")
    parser.add_argument("--bandpass-name", default="custom_J", help="Name to register the custom bandpass under")
    parser.add_argument("--synthetic", action="store_true", help="Create a synthetic filter profile instead of downloading from SVO")
    parser.add_argument("--points", type=int, default=100, help="Number of points in the synthetic filter profile")
    args = parser.parse_args()
    
    # List common filters if requested
    if args.list:
        list_common_filters()
        return 0
    
    # Run the custom bandpass example if requested
    if args.example:
        return run_custom_bandpass_example(args.filter, args.bandpass_name, args.sn, args.output)
    
    # Create a synthetic filter profile if requested
    if args.synthetic:
        # For now, we only support synthetic WFCAM J filter
        filter_name = "WFCAM_J"
        output_file = os.path.join(args.output, f"{filter_name}.dat")
        
        # Create the synthetic filter profile
        wave, trans = create_synthetic_filter(
            WFCAM_J_WAVE_MIN, WFCAM_J_WAVE_MAX, 
            num_points=args.points, 
            filter_name=filter_name
        )
        
        # Save the filter profile
        save_filter_data(wave, trans, output_file, filter_name=filter_name)
        
        print(f"\nSynthetic {filter_name} filter profile created and saved to {output_file}")
        print("You can use this filter profile in JAX-bandflux by including it in your settings.yaml file.")
        
        return 0
    
    # Download the filter profile
    try:
        wave, trans, success = download_svo_filter(args.filter, args.output, args.force)
        if not success:
            print(f"Failed to download {args.filter} filter profile")
            return 1
            
        print(f"Successfully downloaded {args.filter} filter profile")
        print(f"Wavelength range: {wave.min():.2f} - {wave.max():.2f} Angstroms")
        print(f"Number of data points: {len(wave)}")
        
        # Special note for WFCAM J filter
        if args.filter == "UKIRT/WFCAM.J":
            print("\nIMPORTANT: The J bandpass is now ready to use in the JAX-bandflux codebase.")
            print("You can include it in your selected_bandpasses list in settings.yaml.")
            print("Various J-band variants (J, J_1D3, etc.) will use this filter profile.")
            
            # Add a note about the J_1D3 designation
            with open(os.path.join(args.output, "J_BANDPASS_NOTE.txt"), "w") as f:
                f.write("NOTE: The WFCAM J bandpass file (UKIRT_WFCAM.J.dat) is used for all J-band variants.\n")
                f.write("Variants like J_1D3 refer to specific detector or readout channels in the WFCAM instrument.\n")
                f.write("They are not different filters from the standard WFCAM J filter.\n")
                f.write("For photometric analysis, using the standard WFCAM J filter profile is appropriate for all variants.\n")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 