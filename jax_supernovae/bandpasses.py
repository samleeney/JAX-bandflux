"""Bandpass handling for JAX supernova models."""
import os
import jax.numpy as jnp
import numpy as np
from functools import partial
import math
from jax_supernovae.utils import interp
from jax_supernovae.constants import HC_ERG_AA, C_AA_PER_S, MODEL_BANDFLUX_SPACING

# Get package directory
PACKAGE_DIR = os.path.dirname(__file__)

class Bandpass:
    """Bandpass filter class."""
    
    def __init__(self, wave, trans, integration_spacing=MODEL_BANDFLUX_SPACING):
        """Initialize bandpass with wavelength and transmission arrays."""
        self._wave = jnp.asarray(wave)
        self._trans = jnp.asarray(trans)
        self._minwave = float(jnp.min(wave))
        self._maxwave = float(jnp.max(wave))
        
        # Pre-compute integration grid to match sncosmo exactly
        range_diff = self._maxwave - self._minwave
        n_steps = math.ceil(range_diff / integration_spacing)
        self._integration_spacing = range_diff / n_steps
        
        # Create grid starting at minwave + 0.5 * spacing
        self._integration_wave = jnp.linspace(
            self._minwave + 0.5 * self._integration_spacing,
            self._maxwave - 0.5 * self._integration_spacing,
            n_steps
        )
    
    def __call__(self, wave):
        """Get interpolated transmission at given wavelengths."""
        wave = jnp.asarray(wave)
        return interp(wave, self._wave, self._trans)
    
    def minwave(self):
        """Get minimum wavelength."""
        return self._minwave
    
    def maxwave(self):
        """Get maximum wavelength."""
        return self._maxwave
    
    @property
    def wave(self):
        """Get wavelength array."""
        return self._wave
    
    @property
    def trans(self):
        """Get transmission array."""
        return self._trans
        
    @property
    def integration_wave(self):
        """Get pre-computed integration wavelength grid."""
        return self._integration_wave
        
    @property
    def integration_spacing(self):
        """Get integration grid spacing."""
        return self._integration_spacing

# Registry to store bandpasses
_BANDPASSES = {}

def get_bandpass_filepath(band):
    """Map bandpass name to file path.
    
    Parameters
    ----------
    band : str
        Bandpass name (e.g., 'c', 'o', 'ztfg', 'g', etc.)
        
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
        
        # SDSS bandpasses
        'g': 'bandpasses/sdss/sdss_g.dat',  # SDSS g-band
        'r': 'bandpasses/sdss/sdss_r.dat',  # SDSS r-band
        'i': 'bandpasses/sdss/sdss_i.dat',  # SDSS i-band
        'z': 'bandpasses/sdss/sdss_z.dat',  # SDSS z-band
        
        # 2MASS bandpasses
        'H': 'bandpasses/2mass/2mass.H',    # 2MASS H-band
    }
    
    if band not in bandpass_map:
        raise ValueError(f"Unknown bandpass: {band}. Available bandpasses: {list(bandpass_map.keys())}")
    
    # Look for the file in sncosmo-modelfiles directory
    filepath = os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles', bandpass_map[band])
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
            # All other files are simple two-column format
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

def register_all_bandpasses():
    """Register bandpasses in JAX and return dictionaries of bandpasses and bridges.
    
    Returns
    -------
    tuple
        A tuple containing:
        - bandpass_dict: Dictionary mapping bandpass names to Bandpass objects
        - bridges_dict: Dictionary mapping bandpass names to precomputed bridge data
    """
    from jax_supernovae.salt3 import precompute_bandflux_bridge
    
    bandpass_info = [
        # ZTF bandpasses
        {'name': 'ztfg', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/ztf/P48_g.dat'), 'skiprows': 1},
        {'name': 'ztfr', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/ztf/P48_R.dat'), 'skiprows': 1},
        
        # ATLAS bandpasses
        {'name': 'c', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/atlas/Atlas.Cyan'), 'skiprows': 0},
        {'name': 'o', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/atlas/Atlas.Orange'), 'skiprows': 0},
        
        # SDSS bandpasses
        {'name': 'g', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/sdss/sdss_g.dat'), 'skiprows': 0},
        {'name': 'r', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/sdss/sdss_r.dat'), 'skiprows': 0},
        {'name': 'i', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/sdss/sdss_i.dat'), 'skiprows': 0},
        {'name': 'z', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/sdss/sdss_z.dat'), 'skiprows': 0},
        
        # 2MASS bandpasses
        {'name': 'H', 'file': os.path.join(PACKAGE_DIR, 'sncosmo-modelfiles/bandpasses/2mass/2mass.H'), 'skiprows': 0},
    ]
    
    bandpass_dict = {}
    bridges_dict = {}
    for info in bandpass_info:
        try:
            data = np.loadtxt(info['file'], skiprows=info['skiprows'])
            wave, trans = data[:, 0], data[:, 1]
            jax_bandpass = Bandpass(wave, trans)
            register_bandpass(info['name'], jax_bandpass, force=True)
            bandpass_dict[info['name']] = jax_bandpass
            bridges_dict[info['name']] = precompute_bandflux_bridge(jax_bandpass)
        except Exception as e:
            print(f"Warning: Failed to load bandpass {info['name']}: {e}")
    
    return bandpass_dict, bridges_dict 