"""Bandpass handling for JAX supernova models."""
import os
import jax.numpy as jnp
import numpy as np
from functools import partial
import math
from jax_supernovae.utils import interp
from jax_supernovae.constants import HC_ERG_AA, C_AA_PER_S, MODEL_BANDFLUX_SPACING
import requests

# Get package directory
PACKAGE_DIR = os.path.dirname(__file__)

class Bandpass:
    """Bandpass filter class."""
    
    def __init__(self, wave, trans, integration_spacing=MODEL_BANDFLUX_SPACING, name=None):
        """Initialize bandpass with wavelength and transmission arrays."""
        self._wave = jnp.asarray(wave)
        self._trans = jnp.asarray(trans)
        self._name = name
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
    
    def __call__(self, wave, shift=0.0):
        """Get interpolated transmission at given wavelengths with optional shift.
        
        Parameters
        ----------
        wave : array_like
            Wavelengths at which to evaluate transmission
        shift : float, optional
            Constant wavelength shift to apply (in Angstroms)
        """
        wave = jnp.asarray(wave)
        
        # Apply constant shift
        effective_wave = wave - shift
            
        return interp(effective_wave, self._wave, self._trans)

    @property
    def name(self):
        """Optional human-readable bandpass name."""
        return self._name
    
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
            trans=jnp.array(data[:, 1]),
            name=band
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Bandpass file for '{band}' not found at {fname}")
    except Exception as e:
        raise ValueError(f"Error loading bandpass file for '{band}': {e}")

def load_bandpass_from_file(filepath, skiprows=0, name=None):
    """Load a bandpass from a custom file path.
    
    Parameters
    ----------
    filepath : str
        Path to the bandpass file
    skiprows : int, optional
        Number of header rows to skip
    name : str, optional
        Name to register the bandpass under. If None, uses the filename.
        
    Returns
    -------
    tuple
        (name, bandpass) where:
        - name is the registered name of the bandpass
        - bandpass is the Bandpass object
    """
    try:
        # If name not provided, use filename without extension
        if name is None:
            name = os.path.splitext(os.path.basename(filepath))[0]
            
        # Load data from file
        data = np.loadtxt(filepath, skiprows=skiprows)
        
        # Create bandpass object
        bandpass = Bandpass(
            wave=jnp.array(data[:, 0]),
            trans=jnp.array(data[:, 1]),
            name=name
        )
        
        return name, bandpass
    except FileNotFoundError:
        raise FileNotFoundError(f"Bandpass file not found at {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading bandpass file from {filepath}: {e}")

def create_bandpass_from_svo(filter_id, output_dir='filter_data', force_download=False):
    """Create a bandpass object from a filter profile in the SVO Filter Profile Service.
    
    This function downloads a filter profile from the Spanish Virtual Observatory (SVO)
    Filter Profile Service and creates a Bandpass object from it.
    
    Parameters
    ----------
    filter_id : str
        The SVO filter identifier, e.g., 'UKIRT/WFCAM.J'
    output_dir : str, optional
        Directory to save the downloaded filter file. Default is 'filter_data'.
    force_download : bool, optional
        If True, download the filter even if it already exists locally.
    
    Returns
    -------
    Bandpass
        A Bandpass object for the specified filter
    
    Raises
    ------
    FileNotFoundError
        If the filter profile file cannot be found or downloaded
    """
    # Check if the file already exists locally
    local_filename = os.path.join(output_dir, f"{filter_id.replace('/', '_')}.dat")
    if os.path.exists(local_filename) and not force_download:
        try:
            print(f"Loading filter {filter_id} from local file {local_filename}")
            data = np.loadtxt(local_filename)
            wave, trans = data[:, 0], data[:, 1]
            return Bandpass(wave=jnp.array(wave), trans=jnp.array(trans), name=filter_id)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load filter profile from {local_filename}: {e}")
    else:
        # If the file doesn't exist locally, suggest using the download_svo_filter.py script
        raise FileNotFoundError(
            f"Filter profile file for {filter_id} not found at {local_filename}. "
            f"Please use the download_svo_filter.py script to download it:\n"
            f"python examples/download_svo_filter.py --filter {filter_id}"
        )

def register_bandpass(name, bandpass, force=False):
    """Register a bandpass with a given name.
    
    Parameters
    ----------
    name : str
        Name to register the bandpass under
    bandpass : Bandpass
        Bandpass object to register
    force : bool, optional
        If True, overwrite any existing bandpass with the same name
        
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If a bandpass with the given name already exists and force=False
    """
    global _BANDPASSES
    
    if name in _BANDPASSES and not force:
        raise ValueError(f"Bandpass '{name}' already exists in registry")
    
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

    Notes
    -----
    Bandpasses must be registered before use. Common bands (Bessell, SDSS, etc.)
    are automatically registered when SALT3Source is initialized. For custom bands,
    use register_bandpass() or register_all_bandpasses() before JIT compilation.
    """
    if isinstance(name, Bandpass):
        return name
    if name not in _BANDPASSES:
        raise ValueError(
            f"Bandpass '{name}' not found in registry. "
            f"Available bandpasses: {list(_BANDPASSES.keys())}. "
            f"Bandpasses must be registered before use, especially before JIT compilation."
        )
    return _BANDPASSES[name]

def load_custom_bandpasses(bandpass_files):
    """Load custom bandpasses from a list of file paths.
    
    Parameters
    ----------
    bandpass_files : list of str or dict
        List of file paths to bandpass files, or a dictionary mapping
        bandpass names to file paths
        
    Returns
    -------
    dict
        Dictionary mapping bandpass names to Bandpass objects
    """
    custom_bandpasses = {}
    
    if not bandpass_files:
        return custom_bandpasses
        
    if isinstance(bandpass_files, dict):
        # Dictionary mapping names to file paths
        for name, filepath in bandpass_files.items():
            try:
                _, bandpass = load_bandpass_from_file(filepath, name=name)
                register_bandpass(name, bandpass, force=True)
                custom_bandpasses[name] = bandpass
                print(f"Registered custom bandpass '{name}' from {filepath}")
            except Exception as e:
                print(f"Warning: Failed to load custom bandpass '{name}' from {filepath}: {e}")
    else:
        # List of file paths
        for filepath in bandpass_files:
            try:
                name, bandpass = load_bandpass_from_file(filepath)
                register_bandpass(name, bandpass, force=True)
                custom_bandpasses[name] = bandpass
                print(f"Registered custom bandpass '{name}' from {filepath}")
            except Exception as e:
                print(f"Warning: Failed to load custom bandpass from {filepath}: {e}")
    
    return custom_bandpasses

def register_all_bandpasses(custom_bandpass_files=None, svo_filters=None):
    """Register bandpasses in JAX and return dictionaries of bandpasses and bridges.
    
    Parameters
    ----------
    custom_bandpass_files : list or dict, optional
        List of file paths to custom bandpass files, or a dictionary mapping
        bandpass names to file paths
    svo_filters : list, optional
        List of dictionaries containing SVO filter information. Each dictionary
        should have the following keys:
        - 'name': Name to register the bandpass under
        - 'filter_id': SVO filter identifier (e.g., 'UKIRT/WFCAM.J')
        - 'variants': Optional list of variant names to register using the same bandpass
    
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

    # Load commonly used bands from sncosmo (Bessell filters)
    sncosmo_bands = ['bessellb', 'bessellv', 'bessellr', 'besselli', 'bessellux']
    
    bandpass_dict = {}
    bridges_dict = {}
    
    # Load standard bandpasses
    for info in bandpass_info:
        try:
            data = np.loadtxt(info['file'], skiprows=info['skiprows'])
            wave, trans = data[:, 0], data[:, 1]
            jax_bandpass = Bandpass(wave, trans, name=info['name'])
            register_bandpass(info['name'], jax_bandpass, force=True)
            bandpass_dict[info['name']] = jax_bandpass
            bridges_dict[info['name']] = precompute_bandflux_bridge(jax_bandpass)
        except Exception as e:
            print(f"Warning: Failed to load bandpass {info['name']}: {e}")

    # Load Bessell and other common bands from sncosmo
    try:
        import sncosmo
        for band_name in sncosmo_bands:
            try:
                snc_bandpass = sncosmo.get_bandpass(band_name)
                jax_bandpass = Bandpass(snc_bandpass.wave, snc_bandpass.trans, name=band_name)
                register_bandpass(band_name, jax_bandpass, force=True)
                bandpass_dict[band_name] = jax_bandpass
                bridges_dict[band_name] = precompute_bandflux_bridge(jax_bandpass)
            except Exception as e:
                print(f"Warning: Failed to load bandpass {band_name} from sncosmo: {e}")
    except ImportError:
        print("Warning: sncosmo not available, skipping Bessell filter registration")
    
    # Load SVO filter bandpasses if provided
    if svo_filters:
        for filter_info in svo_filters:
            try:
                # Create and register the main bandpass
                bandpass = create_bandpass_from_svo(filter_info['filter_id'])
                
                # Register the main bandpass
                register_bandpass(filter_info['name'], bandpass, force=True)
                bandpass_dict[filter_info['name']] = bandpass
                bridges_dict[filter_info['name']] = precompute_bandflux_bridge(bandpass)
                print(f"Registered {filter_info['name']} bandpass from SVO Filter Profile Service")
                
                # Register variants if any
                if 'variants' in filter_info and filter_info['variants']:
                    for variant in filter_info['variants']:
                        register_bandpass(variant, bandpass, force=True)
                        bandpass_dict[variant] = bandpass
                        bridges_dict[variant] = precompute_bandflux_bridge(bandpass)
                    print(f"Registered {len(filter_info['variants'])} variants of {filter_info['name']} bandpass")
            except Exception as e:
                print(f"Warning: Failed to create {filter_info['name']} bandpass from SVO: {e}")
    
    # Load custom bandpasses if provided
    if custom_bandpass_files:
        custom_bandpasses = load_custom_bandpasses(custom_bandpass_files)
        for name, bandpass in custom_bandpasses.items():
            bandpass_dict[name] = bandpass
            bridges_dict[name] = precompute_bandflux_bridge(bandpass)
    
    return bandpass_dict, bridges_dict 
