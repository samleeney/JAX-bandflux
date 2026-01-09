Bandpass Loading
================

JAX-bandflux provides flexible bandpass management for working with
astronomical filters from various sources.

Built-in Bandpasses
-------------------

JAX-bandflux includes common astronomical filters:

.. testcode::

   from jax_supernovae.bandpasses import get_bandpass, Bandpass, register_bandpass
   from jax_supernovae.salt3 import precompute_bandflux_bridge

   # Access a built-in bandpass
   bp_b = get_bandpass('bessellb')
   print(f"Bessell B: {bp_b.minwave():.0f} - {bp_b.maxwave():.0f} Angstroms")

   # Compare wavelength coverage across filters
   for name in ['bessellb', 'bessellv', 'bessellr', 'besselli']:
       bp = get_bandpass(name)
       print(f"  {name:10s}: {bp.minwave():7.1f} - {bp.maxwave():7.1f} A")


.. testoutput::

   Bessell B: 3600 - 5600 Angstroms
     bessellb  :  3600.0 -  5600.0 A
     bessellv  :  4700.0 -  7000.0 A
     bessellr  :  5500.0 -  9000.0 A
     besselli  :  7000.0 -  9200.0 A

Available built-in bandpasses:

- **Bessell**: ``bessellb``, ``bessellv``, ``bessellr``, ``besselli``, ``bessellux``
- **SDSS**: ``g``, ``r``, ``i``, ``z``
- **ZTF**: ``ztfg``, ``ztfr``
- **ATLAS**: ``c``, ``o``
- **2MASS**: ``H``

Creating Custom Bandpasses
--------------------------

Create a bandpass from wavelength and transmission arrays:

.. testcode::

   # Define a Gaussian bandpass
   wavelengths = np.linspace(4000, 5000, 100)
   transmission = np.exp(-((wavelengths - 4500) / 200)**2)

   # Create Bandpass object
   bandpass = Bandpass(wavelengths, transmission, name='custom_g')
   print(f"Custom bandpass '{bandpass.name}':")
   print(f"  Wavelength range: {bandpass.minwave():.1f} - {bandpass.maxwave():.1f} A")
   print(f"  Peak wavelength: ~4500 A (Gaussian center)")
   print(f"  Number of points: {len(wavelengths)}")


.. testoutput::

   Custom bandpass 'custom_g':
     Wavelength range: 4000.0 - 5000.0 A
   Peak wavelength: ~4500 A (Gaussian center)
     Number of points: 100

The ``Bandpass`` class automatically:

- Normalizes transmission values
- Creates integration grids for flux calculations
- Provides interpolation for arbitrary wavelengths

Registering Bandpasses
~~~~~~~~~~~~~~~~~~~~~~

Register custom bandpasses for use by name:

.. testcode::

   # Register the bandpass
   register_bandpass('my_filter', bandpass, force=True)

   # Now accessible by name
   retrieved = get_bandpass('my_filter')
   print(f"Retrieved '{retrieved.name}' as 'my_filter'")
   print(f"Range: {retrieved.minwave():.1f} - {retrieved.maxwave():.1f} A")


.. testoutput::

   Retrieved 'custom_g' as 'my_filter'
   Range: 4000.0 - 5000.0 A

Loading from Files
------------------

Load bandpass data from text files:

.. code-block:: python

   from jax_supernovae.bandpasses import load_bandpass_from_file

   # Load from two-column file (wavelength, transmission)
   bandpass = load_bandpass_from_file('my_filter.dat', name='my_filter')

   # Skip header rows if needed
   bandpass = load_bandpass_from_file('my_filter.dat', skiprows=1, name='my_filter')

Expected file format:

.. code-block:: text

   # wavelength(Å)  transmission
   4000.0  0.001
   4050.0  0.050
   4100.0  0.200
   ...

Spanish Virtual Observatory (SVO)
---------------------------------

Download filters from the SVO Filter Profile Service:

.. code-block:: python

   from jax_supernovae.bandpasses import create_bandpass_from_svo

   # Download UKIRT J-band filter
   bandpass = create_bandpass_from_svo('UKIRT/WFCAM.J', output_dir='filter_data')

   # Register for later use
   register_bandpass('ukirt_j', bandpass)

The SVO provides thousands of filters from major observatories and instruments.
Browse available filters at: http://svo2.cab.inta-csic.es/svo/theory/fps/

Pre-computing Bridges
---------------------

For high-performance calculations, pre-compute integration bridges:

.. testcode::

   # Pre-compute bridge for a bandpass
   bridge = precompute_bandflux_bridge(bandpass)
   print(f"Bridge structure keys: {sorted(bridge.keys())}")
   print(f"Wavelength grid: {bridge['wave'].shape[0]} points")
   print(f"Grid spacing: {bridge['dwave']} Angstroms")
   print(f"Wavelength range: {float(bridge['wave'][0]):.1f} - {float(bridge['wave'][-1]):.1f} A")


.. testoutput::

   Bridge structure keys: ['dwave', 'trans', 'trans_original', 'wave', 'wave_original', 'zpbandflux_ab']
   Wavelength grid: 200 points
   Grid spacing: 5.0 Angstroms
   Wavelength range: 4002.5 - 4997.5 A

Bridges contain:

- ``wave``: Integration wavelength grid
- ``trans``: Transmission values on the grid
- ``dwave``: Grid spacing (typically 5.0 Å)

Multiple Bandpasses
~~~~~~~~~~~~~~~~~~~

Pre-compute bridges for all bands in your dataset:

.. testcode::

   # Define unique bands
   unique_bands = ['bessellb', 'bessellv', 'bessellr']

   # Pre-compute all bridges
   bridges = tuple(precompute_bandflux_bridge(get_bandpass(b))
                   for b in unique_bands)
   print(f"Pre-computed {len(bridges)} bridges")

   # Inspect bridge sizes
   for band, bridge in zip(unique_bands, bridges):
       print(f"  {band}: {bridge['wave'].shape[0]} points, {float(bridge['wave'][0]):.0f}-{float(bridge['wave'][-1]):.0f} A")

   # Create band index mapping
   band_to_idx = {b: i for i, b in enumerate(unique_bands)}
   print(f"Band index mapping: {band_to_idx}")


.. testoutput::

   Pre-computed 3 bridges
     bessellb: 400 points, 3602-5598 A
     bessellv: 460 points, 4702-6998 A
     bessellr: 700 points, 5502-8998 A
   Band index mapping: {'bessellb': 0, 'bessellv': 1, 'bessellr': 2}

Bandpass Properties
-------------------

Query bandpass characteristics:

.. testcode::

   bp = get_bandpass('bessellv')

   # Wavelength range
   print(f"Bessell V properties:")
   print(f"  Wavelength range: {bp.minwave():.1f} - {bp.maxwave():.1f} A")
   print(f"  Central wavelength: ~{(bp.minwave() + bp.maxwave()) / 2:.0f} A")

   # Access raw data
   print(f"  Data points: {len(bp.wave)} wavelengths, {len(bp.trans)} transmission values")
   print(f"  Peak transmission: {float(max(bp.trans)):.3f}")


.. testoutput::

   Bessell V properties:
     Wavelength range: 4700.0 - 7000.0 A
     Central wavelength: ~5850 A
     Data points: 24 wavelengths, 24 transmission values
     Peak transmission: 1.000

Interpolation
~~~~~~~~~~~~~

Bandpasses support interpolation at arbitrary wavelengths:

.. testcode::

   # Get transmission at specific wavelengths
   wave_query = np.array([5000.0, 5500.0, 6000.0, 6500.0])
   trans_values = bp(wave_query)
   print("Bessell V transmission:")
   for w, t in zip(wave_query, trans_values):
       print(f"  {w:.0f} A: {float(t):.3f}")


.. testoutput::

   Bessell V transmission:
     5000 A: 0.485
     5500 A: 0.865
     6000 A: 0.317
     6500 A: 0.037

Wavelength Shifts
-----------------

Apply wavelength shifts to bandpasses (useful for filter calibration):

.. testcode::

   # Get transmission with different wavelength shifts
   wave_query = np.array([5500.0])
   print("Effect of wavelength shift on transmission at 5500 A:")
   for shift in [-20.0, -10.0, 0.0, 10.0, 20.0]:
       trans = bp(wave_query, shift=shift)
       print(f"  Shift {shift:+5.1f} A: transmission = {float(trans[0]):.4f}")


.. testoutput::

   Effect of wavelength shift on transmission at 5500 A:
     Shift -20.0 A: transmission = 0.8422
     Shift -10.0 A: transmission = 0.8538
     Shift  +0.0 A: transmission = 0.8653
     Shift +10.0 A: transmission = 0.8743
     Shift +20.0 A: transmission = 0.8833

Registering All Standard Bandpasses
-----------------------------------

Register all standard bandpasses at once:

.. code-block:: python

   from jax_supernovae.bandpasses import register_all_bandpasses

   # Register standard and custom bandpasses
   bandpass_dict, bridges_dict = register_all_bandpasses(
       custom_bandpass_files={'my_filter': 'path/to/filter.dat'},
       svo_filters={'jwst_f150w': 'JWST/NIRCam.F150W'}
   )

   print(f"Registered {len(bandpass_dict)} bandpasses")

Best Practices
--------------

1. **Pre-compute bridges once**: Do this outside your likelihood function

   .. code-block:: python

      # GOOD: Pre-compute once
      bridges = tuple(precompute_bandflux_bridge(get_bandpass(b))
                      for b in unique_bands)

      @jax.jit
      def likelihood(params):
          return source.bandflux(params, None, phases, bridges=bridges, ...)

2. **Use consistent naming**: Register custom bandpasses with descriptive names

3. **Cache SVO downloads**: Use ``output_dir`` to save downloaded filters locally

4. **Check wavelength coverage**: Ensure your bandpass covers the relevant wavelength
   range for your supernova model

   .. testcode::

      source = SALT3Source()
      bp = get_bandpass('bessellv')

      # Verify bandpass is within model range
      print(f"Model range: {source.minwave():.0f} - {source.maxwave():.0f} A")
      print(f"Bandpass range: {bp.minwave():.0f} - {bp.maxwave():.0f} A")
      print(f"Bandpass within model? {bp.minwave() > source.minwave() and bp.maxwave() < source.maxwave()}")

   .. testoutput::

      Model range: 2000 - 20000 A
      Bandpass range: 4700 - 7000 A
      Bandpass within model? True
