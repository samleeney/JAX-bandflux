Dust Extinction
===============

JAX-bandflux includes support for Milky Way dust extinction models that can be applied to supernova spectra. Three standard extinction laws are implemented: CCM89, O'Donnell 1994, and Fitzpatrick 1999.

Available Dust Laws
-------------------

The following dust extinction laws are available:

* **CCM89** (``dust_type=0``): Cardelli, Clayton & Mathis (1989) - Standard Galactic extinction law
* **OD94** (``dust_type=1``): O'Donnell (1994) - Improved UV extinction
* **F99** (``dust_type=2``): Fitzpatrick (1999) - More flexible parameterization

Basic Usage
-----------

To apply dust extinction to SALT3 model calculations, add dust parameters to your model parameters dictionary:

.. code-block:: python

   import jax.numpy as jnp
   import jax_supernovae as js

   # Load data for a supernova
   times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = js.data.load_and_process_data('19dwz')

   # Define SALT3 parameters with dust extinction
   params = {
       'z': fixed_z[0],
       't0': 58650.0,
       'x0': 1e-5,
       'x1': 0.0,
       'c': 0.0,
       # Dust parameters
       'dust_type': 0,  # CCM89 law
       'ebv': 0.1,      # E(B-V) reddening
       'r_v': 3.1       # Total-to-selective extinction ratio
   }

   # Calculate model fluxes with dust extinction applied
   model_fluxes = js.salt3.optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
   model_fluxes = model_fluxes[jnp.arange(len(times)), band_indices]

Comparing Dust Laws
-------------------

Different dust laws produce different extinction curves, particularly in the UV:

.. code-block:: python

   import jax.numpy as jnp
   from jax_supernovae import dust
   import matplotlib.pyplot as plt

   # Define wavelength grid
   wave = jnp.linspace(2000, 10000, 500)
   ebv = 0.1
   r_v = 3.1

   # Calculate extinction for each law
   ext_ccm89 = dust.ccm89_extinction(wave, ebv, r_v)
   ext_od94 = dust.od94_extinction(wave, ebv, r_v)
   ext_f99 = dust.f99_extinction(wave, ebv, r_v)

   # Plot extinction curves
   plt.figure(figsize=(10, 6))
   plt.plot(wave, ext_ccm89, label='CCM89')
   plt.plot(wave, ext_od94, label='OD94')
   plt.plot(wave, ext_f99, label='F99')
   plt.xlabel('Wavelength (Å)')
   plt.ylabel('A(λ) (mag)')
   plt.legend()
   plt.title(f'Dust Extinction Laws (E(B-V)={ebv}, R_V={r_v})')
   plt.show()

Effect on Light Curves
----------------------

Dust extinction affects blue bands more than red bands, changing both the color and amplitude of light curves:

.. code-block:: python

   import jax.numpy as jnp
   import jax_supernovae as js

   # Parameters without dust
   params_nodust = {'z': 0.1, 't0': 0.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0}

   # Parameters with moderate dust
   params_dust = params_nodust.copy()
   params_dust.update({'dust_type': 0, 'ebv': 0.2, 'r_v': 3.1})

   # Calculate fluxes for comparison
   phases = jnp.linspace(-20, 50, 100)

   # Using a blue bandpass (more affected)
   flux_nodust_blue = js.salt3.salt3_bandflux(phases, blue_bandpass, params_nodust)
   flux_dust_blue = js.salt3.salt3_bandflux(phases, blue_bandpass, params_dust)

   # Using a red bandpass (less affected)
   flux_nodust_red = js.salt3.salt3_bandflux(phases, red_bandpass, params_nodust)
   flux_dust_red = js.salt3.salt3_bandflux(phases, red_bandpass, params_dust)

   # Dust reduces flux more in blue than red
   blue_reduction = flux_dust_blue / flux_nodust_blue  # ~0.6 for E(B-V)=0.2
   red_reduction = flux_dust_red / flux_nodust_red     # ~0.8 for E(B-V)=0.2

Parameter Reference
-------------------

.. list-table:: Dust Extinction Parameters
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type/Range
     - Description
   * - ``dust_type``
     - int (0, 1, 2)
     - Dust law selection: 0=CCM89, 1=OD94, 2=F99
   * - ``ebv``
     - float (≥0)
     - E(B-V) color excess in magnitudes
   * - ``r_v``
     - float (typically 2-5)
     - Ratio of total to selective extinction (default: 3.1)

Notes
-----

* Dust extinction is applied to the rest-frame SED before redshifting
* The implementation matches `sncosmo` extinction laws for consistency
* All dust laws are JIT-compiled for performance
* Gradients flow through dust parameters for optimization