Custom Bandpasses Tutorial
========================

This tutorial demonstrates how to use custom bandpasses with JAX-bandflux.

Prerequisites
-----------

Before starting this tutorial, make sure you have installed JAX-bandflux:

.. code-block:: bash

   pip install jax-bandflux

Built-in Bandpasses
-----------------

JAX-bandflux supports a variety of standard bandpasses out of the box, including:

- ZTF bandpasses: ``ztfg``, ``ztfr``
- ATLAS bandpasses: ``c``, ``o``
- SDSS bandpasses: ``g``, ``r``, ``i``, ``z``
- 2MASS bandpasses: ``H``
- WFCAM bandpasses: ``J``, ``J_1D3``

You can use these bandpasses directly without any additional setup:

.. code-block:: python

   from jax_supernovae.bandpasses import register_all_bandpasses
   
   # Register all built-in bandpasses
   bandpass_dict, bridges_dict = register_all_bandpasses()
   
   # Access a specific bandpass
   ztfg_bandpass = bandpass_dict['ztfg']

Downloading Bandpasses from SVO
-----------------------------

JAX-bandflux includes a utility script to download filter profiles from the Spanish Virtual Observatory (SVO) Filter Profile Service:

.. code-block:: bash

   # Download the WFCAM J filter profile
   python examples/download_svo_filter.py --filter UKIRT/WFCAM.J
   
   # List available common filters
   python examples/download_svo_filter.py --list

You can also use this script programmatically:

.. code-block:: python

   import os
   import sys
   sys.path.append('examples')
   from download_svo_filter import download_filter
   
   # Download a filter profile
   filter_path = download_filter('UKIRT/WFCAM.J', output_dir='filter_data')
   
   # Print the path to the downloaded filter profile
   print(f"Filter profile downloaded to: {filter_path}")

Creating Custom Bandpasses
------------------------

You can create custom bandpasses in several ways:

1. From a file:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from jax_supernovae.bandpasses import Bandpass, register_bandpass, load_bandpass_from_file
   
   # Load a bandpass from a file
   bandpass = load_bandpass_from_file('filter_data/UKIRT_WFCAM.J.dat')
   
   # Register the bandpass
   register_bandpass('my_custom_J', bandpass)

2. From arrays:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from jax_supernovae.bandpasses import Bandpass, register_bandpass
   
   # Create a bandpass from arrays
   wave = jnp.linspace(4000, 5000, 100)  # Wavelength in Angstroms
   trans = jnp.exp(-((wave - 4500) / 200)**2)  # Transmission
   
   # Create a bandpass object
   bandpass = Bandpass(wave=wave, trans=trans)
   
   # Register the bandpass
   register_bandpass('my_gaussian_bandpass', bandpass)

3. From a synthetic profile:

.. code-block:: python

   import sys
   sys.path.append('examples')
   from download_svo_filter import create_synthetic_filter
   from jax_supernovae.bandpasses import load_bandpass_from_file, register_bandpass
   
   # Create a synthetic filter profile
   filter_path = create_synthetic_filter(
       central_wavelength=12350,  # Central wavelength in Angstroms
       fwhm=1545,  # Full width at half maximum in Angstroms
       points=100,  # Number of points in the profile
       output_file='filter_data/synthetic_J.dat'  # Output file path
   )
   
   # Load the synthetic bandpass
   bandpass = load_bandpass_from_file(filter_path)
   
   # Register the bandpass
   register_bandpass('synthetic_J', bandpass)

Using Custom Bandpasses in SALT3 Model Fitting
-------------------------------------------

Once you have registered your custom bandpasses, you can use them in SALT3 model fitting:

.. code-block:: python

   import jax.numpy as jnp
   from jax_supernovae.salt3 import salt3_bandflux, precompute_bandflux_bridge
   
   # Precompute bridge data for efficient flux calculations
   bridge = precompute_bandflux_bridge(bandpass)
   
   # Define SALT3 parameters
   params = {
       'z': 0.1,
       't0': 0.0,
       'x0': 1e-5,
       'x1': 0.0,
       'c': 0.0
   }
   
   # Compute bandflux
   time = 0.0
   flux = salt3_bandflux(time, bridge, params)
   
   print(f"Flux: {float(flux):.6e}")

Complete Example
--------------

Here's a complete example that demonstrates how to download a filter profile from SVO, create a custom bandpass, and use it in SALT3 model fitting:

.. code-block:: python

   import os
   import sys
   import numpy as np
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   
   # Add the examples directory to the path
   sys.path.append('examples')
   from download_svo_filter import download_filter
   
   from jax_supernovae.bandpasses import Bandpass, register_bandpass, load_bandpass_from_file
   from jax_supernovae.salt3 import salt3_bandflux, precompute_bandflux_bridge
   
   # Download a filter profile from SVO
   filter_path = download_filter('UKIRT/WFCAM.J', output_dir='filter_data')
   
   # Load the bandpass
   bandpass = load_bandpass_from_file(filter_path)
   
   # Register the bandpass
   register_bandpass('WFCAM_J', bandpass)
   
   # Precompute bridge data for efficient flux calculations
   bridge = precompute_bandflux_bridge(bandpass)
   
   # Define SALT3 parameters
   params = {
       'z': 0.1,
       't0': 0.0,
       'x0': 1e-5,
       'x1': 0.0,
       'c': 0.0
   }
   
   # Compute bandflux at different times
   times = jnp.linspace(-10, 30, 100)
   fluxes = jnp.array([float(salt3_bandflux(t, bridge, params)) for t in times])
   
   # Plot the light curve
   plt.figure(figsize=(10, 6))
   plt.plot(times, fluxes)
   plt.xlabel('Time (days)')
   plt.ylabel('Flux')
   plt.title('SALT3 Light Curve with WFCAM J Bandpass')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()
   
   # Plot the bandpass
   plt.figure(figsize=(10, 6))
   plt.plot(bandpass.wave, bandpass.trans)
   plt.xlabel('Wavelength (Angstroms)')
   plt.ylabel('Transmission')
   plt.title('WFCAM J Bandpass')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()