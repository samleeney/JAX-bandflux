import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import sncosmo
from jax_supernovae.salt3nir import salt3nir_multiband_flux
from jax_supernovae.core import Bandpass
from jax_supernovae.bandpasses import register_bandpass, get_bandpass
from load_hsf_data import load_hsf_data

# Load HSF data first to see what bands we need
data = load_hsf_data('22nbn')
print("\nUnique bands in data:", np.unique(data['band']))

def register_all_bandpasses():
    """Register bandpasses in both SNCosmo and JAX."""
    # Define the bandpasses to test
    bandpass_info = [
        {'name': 'ztfg', 'file': 'sncosmo-modelfiles/bandpasses/ztf/P48_g.dat', 'skiprows': 1},
        {'name': 'ztfr', 'file': 'sncosmo-modelfiles/bandpasses/ztf/P48_R.dat', 'skiprows': 1},
        {'name': 'c', 'file': 'sncosmo-modelfiles/bandpasses/atlas/Atlas.Cyan', 'skiprows': 0},
        {'name': 'o', 'file': 'sncosmo-modelfiles/bandpasses/atlas/Atlas.Orange', 'skiprows': 0},
        {'name': 'J_1D3', 'file': 'sncosmo-modelfiles/bandpasses/jwst/jwst_nircam_f150w.dat', 'skiprows': 0},
        {'name': 'J_2D', 'file': 'sncosmo-modelfiles/bandpasses/jwst/jwst_nircam_f200w.dat', 'skiprows': 0}
    ]
    
    bandpass_dict = {}
    for info in bandpass_info:
        # Load the bandpass data
        try:
            data = np.loadtxt(info['file'], skiprows=info['skiprows'])
            wave, trans = data[:, 0], data[:, 1]
            
            # Register with SNCosmo if not already registered
            try:
                sncosmo.get_bandpass(info['name'])
            except:
                band = sncosmo.Bandpass(wave, trans, name=info['name'])
                sncosmo.register(band)
            
            # Register with JAX and store in dictionary
            jax_bandpass = Bandpass(wave, trans)
            register_bandpass(info['name'], jax_bandpass, force=True)
            bandpass_dict[info['name']] = jax_bandpass
            print(f"Successfully registered bandpass {info['name']}")
        except Exception as e:
            print(f"Failed to register bandpass {info['name']}: {str(e)}")
    
    return bandpass_dict

# Register bandpasses
bandpass_dict = register_all_bandpasses()

# Create SNCosmo model for comparison
source = sncosmo.SALT3Source(modeldir='sncosmo-modelfiles/models/salt3-nir/salt3nir-p22')
model = sncosmo.Model(source=source)

def objective(parameters):
    """Calculate chi-squared between data and model."""
    # Convert parameters to array if scalar
    parameters = np.asarray(parameters)
    
    # Update SNCosmo model parameters
    model.parameters = parameters
    
    # Get unique bands and their indices, skipping bands we don't have bandpasses for
    unique_bands = []
    for band in np.unique(data['band']):
        if band in bandpass_dict:
            unique_bands.append(band)
        else:
            print(f"Skipping band {band} - no bandpass available")
    
    if not unique_bands:
        raise ValueError("No valid bands found!")
        
    band_to_idx = {band: i for i, band in enumerate(unique_bands)}
    
    # Get bandpasses for each unique band
    bandpasses = [bandpass_dict[band] for band in unique_bands]
    
    # Get times and corresponding band indices for observations with valid bands
    valid_mask = np.array([band in bandpass_dict for band in data['band']])
    times = data['time'][valid_mask]
    band_indices = np.array([band_to_idx[band] for band in data['band'][valid_mask]])
    
    # Calculate fluxes for all bands at once
    jax_fluxes = salt3nir_multiband_flux(times, bandpasses, 
                                        {'z': float(parameters[0]), 
                                         't0': float(parameters[1]), 
                                         'x0': float(parameters[2]), 
                                         'x1': float(parameters[3]), 
                                         'c': float(parameters[4])},
                                        zps=data['zp'][valid_mask], zpsys='ab')
    
    # Extract fluxes for each observation's band
    model_fluxes = jax_fluxes[np.arange(len(times)), band_indices]
    
    # Calculate chi-squared using only valid observations
    chi2 = np.sum(((data['flux'][valid_mask] - model_fluxes) / data['fluxerr'][valid_mask])**2)
    
    # Print current parameters and chi-squared
    print(f"Parameters: z={parameters[0]:.3f}, t0={parameters[1]:.1f}, "
          f"x0={parameters[2]:.2e}, x1={parameters[3]:.2f}, c={parameters[4]:.2f}")
    print(f"Chi-squared: {chi2:.2f}")
    
    return float(chi2)

# Starting parameters and bounds
x0 = np.array([0.4, 59765., 1e-4, 0., 0.])  # z, t0, x0, x1, c
bounds = [(0.1, 0.7), (59700., 59800.), (1e-6, 1e-3), (-3., 3.), (-0.3, 0.3)]

# Run minimizer
result = fmin_l_bfgs_b(objective, x0, bounds=bounds, approx_grad=True)

# Print final results
print("\nFinal parameters:")
print(f"z = {result[0][0]:.3f}")
print(f"t0 = {result[0][1]:.1f}")
print(f"x0 = {result[0][2]:.2e}")
print(f"x1 = {result[0][3]:.2f}")
print(f"c = {result[0][4]:.2f}")
print(f"Final chi-squared: {result[1]:.2f}")

# Compare final fluxes
final_params = {'z': result[0][0], 't0': result[0][1], 
                'x0': result[0][2], 'x1': result[0][3], 
                'c': result[0][4]}

# Update SNCosmo model with final parameters
model.parameters = result[0]

# Calculate final fluxes for all unique bands
unique_bands = np.unique(data['band'])
print("\nComparing final fluxes for each band:")
for band in unique_bands:
    if band not in bandpass_dict:
        print(f"\nSkipping band {band} - no bandpass available")
        continue
        
    band_mask = data['band'] == band
    if not np.any(band_mask):
        continue
        
    times = data['time'][band_mask]
    zps = data['zp'][band_mask]
    
    # Calculate fluxes with both implementations
    sncosmo_flux = model.bandflux(band, times, zp=zps, zpsys='ab')
    jax_flux = salt3nir_multiband_flux(times, [bandpass_dict[band]], final_params, 
                                      zps=zps, zpsys='ab')[:, 0]
    
    # Compare results
    ratio = jax_flux / sncosmo_flux
    max_diff = np.max(np.abs(1 - ratio))
    print(f"\nBand: {band}")
    print(f"Max relative difference: {max_diff:.2e}")
    print(f"Mean ratio (JAX/SNCosmo): {np.mean(ratio):.6f}")
    if max_diff > 1e-5:
        print("WARNING: Fluxes differ by more than 1e-5!")