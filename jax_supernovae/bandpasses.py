"""Bandpass data for common filters."""
import jax.numpy as jnp
import numpy as np
import requests
from io import StringIO
from .core import Bandpass
import os
import sncosmo

# SVO Filter Profile Service base URL
SVO_URL = "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php"

def fetch_svo_bandpass(facility, instrument, band):
    """Fetch bandpass data from SVO Filter Profile Service.
    This is a fallback that we're not currently using.
    """
    raise NotImplementedError("SVO fetch is not currently supported")

def load_sdss_bandpass(band):
    """Load SDSS bandpass data from SNCosmo.
    
    Parameters
    ----------
    band : str
        The SDSS band to load (e.g., 'sdssg', 'sdssr', etc.)
    
    Returns
    -------
    Bandpass
        A Bandpass object containing the wavelength and transmission data.
    """
    # Get the bandpass from SNCosmo
    sncosmo_band = sncosmo.get_bandpass(band)
    
    # Extract wavelength and transmission data
    wave = sncosmo_band.wave
    trans = sncosmo_band.trans
    
    print(f"Loaded bandpass for {band} band from SNCosmo")
    print(f"Wave shape: {wave.shape}, min: {wave.min()}, max: {wave.max()}")
    print(f"Trans shape: {trans.shape}, min: {trans.min()}, max: {trans.max()}")
    
    return Bandpass(wave, trans)

def get_bandpass(name):
    """Get a bandpass object by name or return the bandpass if it's already a Bandpass object."""
    # If it's already a Bandpass object, return it
    if isinstance(name, (Bandpass, sncosmo.bandpasses.Bandpass)):
        # If it's a SNCosmo Bandpass, convert it to our Bandpass
        if isinstance(name, sncosmo.bandpasses.Bandpass):
            return Bandpass(name.wave, name.trans)
        return name
    
    # Otherwise, treat it as a name string
    if name.startswith('sdss'):
        return load_sdss_bandpass(name)
    else:
        raise ValueError(f"Unknown bandpass: {name}") 