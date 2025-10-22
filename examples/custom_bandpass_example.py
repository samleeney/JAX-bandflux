"""
Custom Bandpass Loading Example

This example demonstrates how to:
1. Load custom bandpass filters from files
2. Register them with JAX-bandflux
3. Use them for bandflux calculations
"""

import jax.numpy as jnp
import numpy as np
from jax_supernovae import SALT3Source
from jax_supernovae.bandpasses import Bandpass, register_bandpass, get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge
import os

def load_bandpass_from_file(filepath, name):
    """
    Load a bandpass from a two-column text file.

    Parameters
    ----------
    filepath : str
        Path to the bandpass file (wavelength, transmission)
    name : str
        Name to register the bandpass under

    Returns
    -------
    Bandpass
        The loaded bandpass object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bandpass file not found: {filepath}")

    data = np.loadtxt(filepath)
    wave = data[:, 0]  # Wavelength in Angstroms
    trans = data[:, 1]  # Transmission (0-1)

    bandpass = Bandpass(wave, trans)
    register_bandpass(name, bandpass, force=True)
    print(f"Registered bandpass '{name}' from {filepath}")

    return bandpass


def create_synthetic_bandpass(center_wave, width, name):
    """
    Create a synthetic Gaussian bandpass for testing.

    Parameters
    ----------
    center_wave : float
        Central wavelength in Angstroms
    width : float
        Width (sigma) of the Gaussian in Angstroms
    name : str
        Name to register the bandpass under

    Returns
    -------
    Bandpass
        The synthetic bandpass object
    """
    wave = np.linspace(center_wave - 3*width, center_wave + 3*width, 200)
    trans = np.exp(-0.5 * ((wave - center_wave) / width)**2)

    bandpass = Bandpass(wave, trans)
    register_bandpass(name, bandpass, force=True)
    print(f"Created synthetic bandpass '{name}' centered at {center_wave}Å")

    return bandpass


def download_svo_filter(facility, instrument, filter_name, output_dir='filter_data'):
    """
    Download a filter from the SVO Filter Profile Service.

    Parameters
    ----------
    facility : str
        Facility name (e.g., 'UKIRT', 'HST')
    instrument : str
        Instrument name (e.g., 'WFCAM', 'ACS')
    filter_name : str
        Filter name (e.g., 'J', 'F606W')
    output_dir : str
        Directory to save the filter file

    Returns
    -------
    str
        Path to the saved filter file
    """
    import requests

    os.makedirs(output_dir, exist_ok=True)

    # Construct SVO filter ID
    filter_id = f"{facility}/{instrument}.{filter_name}"
    url = f"http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={filter_id}"

    print(f"Downloading {filter_id} from SVO...")
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to download filter: HTTP {response.status_code}")

    # Parse the XML response to extract wavelength and transmission
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.content)

    data_element = root.find('.//{http://www.ivoa.net/xml/VOTable/v1.1}DATA')
    if data_element is None:
        raise RuntimeError("Could not parse filter data from SVO response")

    tabledata = data_element.find('{http://www.ivoa.net/xml/VOTable/v1.1}TABLEDATA')

    wavelengths = []
    transmissions = []
    for tr in tabledata.findall('{http://www.ivoa.net/xml/VOTable/v1.1}TR'):
        tds = tr.findall('{http://www.ivoa.net/xml/VOTable/v1.1}TD')
        wavelengths.append(float(tds[0].text))
        transmissions.append(float(tds[1].text))

    # Save to file
    output_path = os.path.join(output_dir, f"{facility}_{instrument}.{filter_name}.dat")
    np.savetxt(output_path, np.column_stack([wavelengths, transmissions]),
               header=f"{filter_id} from SVO Filter Profile Service")

    print(f"Saved filter to {output_path}")
    return output_path


# Example 1: Load a custom bandpass from file
print("=" * 60)
print("Example 1: Loading custom bandpass from file")
print("=" * 60)

try:
    # Try to load WFCAM J-band filter if it exists
    j_bandpass = load_bandpass_from_file('filter_data/UKIRT_WFCAM.J.dat', 'custom_j')
    print(f"Bandpass wavelength range: {j_bandpass.wave[0]:.1f} - {j_bandpass.wave[-1]:.1f} Å\n")
except FileNotFoundError:
    print("WFCAM J filter not found. Use download_svo_filter() to get it.\n")


# Example 2: Create a synthetic bandpass
print("=" * 60)
print("Example 2: Creating synthetic bandpass")
print("=" * 60)

synthetic_r = create_synthetic_bandpass(center_wave=6250.0, width=500.0, name='synthetic_r')
print(f"Bandpass wavelength range: {synthetic_r.wave[0]:.1f} - {synthetic_r.wave[-1]:.1f} Å\n")


# Example 3: Use custom bandpass for bandflux calculations
print("=" * 60)
print("Example 3: Using custom bandpass for bandflux calculation")
print("=" * 60)

source = SALT3Source()
params = {'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

# Calculate bandflux at several phases
phases = jnp.array([-10.0, 0.0, 10.0, 20.0])

# Precompute bridge for performance
bridge = precompute_bandflux_bridge(synthetic_r)

# Calculate fluxes
print("Phases (days)  |  Flux")
print("-" * 30)
for phase in phases:
    flux = source.bandflux(
        params, 'synthetic_r', phase,
        zp=27.5, zpsys='ab'
    )
    print(f"{phase:14.1f} | {flux:.6e}")

print()


# Example 4: Download filter from SVO (commented out by default)
print("=" * 60)
print("Example 4: Downloading filter from SVO")
print("=" * 60)

print("To download a filter from SVO, uncomment the following code:")
print("""
try:
    filepath = download_svo_filter('UKIRT', 'WFCAM', 'J')
    j_bandpass = load_bandpass_from_file(filepath, 'ukirt_j')
    print(f"Successfully loaded UKIRT J-band filter")
except Exception as e:
    print(f"Failed to download filter: {e}")
""")

print("\n" + "=" * 60)
print("Custom bandpass example complete!")
print("=" * 60)
