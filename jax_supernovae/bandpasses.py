"""Bandpass handling for JAX supernova models."""
import os
import jax.numpy as jnp
import numpy as np
from jax_supernovae.core import Bandpass

# Registry to store bandpasses
_BANDPASSES = {}

def get_bandpass_filepath(band):
    """Map bandpass name to file path.
    
    Parameters
    ----------
    band : str
        Bandpass name (e.g., 'c', 'o', 'ztfg')
        
    Returns
    -------
    str
        Path to the bandpass file
    """
    bandpass_map = {
        # ATLAS bandpasses
        'c': 'bandpasses/atlas/Atlas.Cyan',
        'o': 'bandpasses/atlas/Atlas.Orange',
        # ZTF bandpasses
        'ztfg': 'bandpasses/ztf/P48_g.dat',
        'ztfr': 'bandpasses/ztf/P48_R.dat',
    }
    
    if band not in bandpass_map:
        raise ValueError(f"Unknown bandpass: {band}. Available bandpasses: {list(bandpass_map.keys())}")
    
    # Look for the file in sncosmo-modelfiles directory
    filepath = os.path.join('sncosmo-modelfiles', bandpass_map[band])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bandpass file not found: {filepath}")
        
    return filepath

def load_bandpass(band):
    """Load a bandpass from file.
    
    Parameters
    ----------
    band : str
        Name of the bandpass to load
        
    Returns
    -------
    bandpass : Bandpass
        A Bandpass object containing the filter transmission curve.
    """
    fname = get_bandpass_filepath(band)
    try:
        # Handle different file formats
        if band in ['ztfg', 'ztfr']:
            # ZTF files have a header line
            data = np.loadtxt(fname, skiprows=1)
        else:
            # ATLAS files are simple two-column format
            data = np.loadtxt(fname)
            
        # Create bandpass object
        return Bandpass(
            wave=jnp.array(data[:, 0]),
            trans=jnp.array(data[:, 1])
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Bandpass file for '{band}' not found at {fname}")
    except Exception as e:
        raise ValueError(f"Error loading bandpass file for '{band}': {e}")

def register_bandpass(name, bandpass, force=False):
    """Register a bandpass in the registry.
    
    Parameters
    ----------
    name : str
        Name to register the bandpass under
    bandpass : Bandpass
        The bandpass object to register
    force : bool, optional
        Whether to overwrite an existing bandpass with the same name
    """
    if name in _BANDPASSES and not force:
        raise ValueError(f"Bandpass {name} already exists")
    _BANDPASSES[name] = bandpass

def get_bandpass(name):
    """Get a bandpass from the registry.
    
    Parameters
    ----------
    name : str or Bandpass
        Name of the bandpass or a Bandpass object
        
    Returns
    -------
    bandpass : Bandpass
        The requested bandpass
    """
    if isinstance(name, Bandpass):
        return name
    if name not in _BANDPASSES:
        raise ValueError(f"Bandpass {name} not found")
    return _BANDPASSES[name]

def register_hsf_bandpasses():
    """Register all bandpasses used in the HSF_DR1 dataset."""
    # Load ATLAS filters
    c_band = load_bandpass('c')
    register_bandpass('c', c_band, force=True)
    
    o_band = load_bandpass('o')
    register_bandpass('o', o_band, force=True)
    
    # Load ZTF filters
    for band in ['ztfg', 'ztfr']:
        bandpass = load_bandpass(band)
        register_bandpass(band, bandpass, force=True) 