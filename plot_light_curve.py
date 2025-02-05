import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
import yaml
from jax_supernovae.data import load_and_process_data
from jax_supernovae.salt3nir import optimized_salt3nir_multiband_flux

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Load settings
with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

# Load data
fix_z = settings.get('fix_z', False)
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz', data_dir='data', fix_z=fix_z)

# Load chains and get best fit parameters
param_names = ['t0', 'x0', 'x1', 'c'] if fix_z else ['z', 't0', 'x0', 'x1', 'c']
samples = read_chains('chains/chains', columns=param_names)

# Get mean parameters as our best fit
best_fit_params = {}
for param in param_names:
    best_fit_params[param] = float(samples[param].mean())
    
# If using fixed redshift, add it to parameters
if fix_z:
    best_fit_params['z'] = fixed_z[0]

# Convert x0 from log space back to linear
best_fit_params['x0'] = 10**best_fit_params['x0']

# Create time grid for smooth model curves
t_min = np.min(times) - 5
t_max = np.max(times) + 5
t_grid = np.linspace(t_min, t_max, 100)

# Get unique bands
unique_bands = np.unique(band_indices)
n_bands = len(unique_bands)

# Set up the plot
plt.figure(figsize=(12, 8))

# Define colours for each band
colours = ['g', 'c', 'orange', 'r']  # g, c, o, r bands
markers = ['o', 's', 'D', '^']

# Plot data points for each band
for i, band_idx in enumerate(unique_bands):
    mask = band_indices == band_idx
    plt.errorbar(times[mask], fluxes[mask], yerr=fluxerrs[mask],
                fmt=markers[i], color=colours[i], label=f'Band {i} Data',
                markersize=8, alpha=0.6)

# Calculate and plot model curves
for i in range(n_bands):
    # Create band indices array for this band
    band_idx_grid = np.full_like(t_grid, i, dtype=int)
    
    # Calculate model fluxes
    model_fluxes = optimized_salt3nir_multiband_flux(
        jnp.array(t_grid),
        bridges,
        best_fit_params,
        zps=zps,
        zpsys='ab'
    )
    
    # Extract fluxes for this band
    band_fluxes = model_fluxes[:, i]
    
    # Plot model curve
    plt.plot(t_grid, band_fluxes, '-', color=colours[i], 
             label=f'Band {i} Model', linewidth=2, alpha=0.8)

# Add labels and title
plt.xlabel('MJD', fontsize=12)
plt.ylabel('Flux', fontsize=12)
title = 'Light Curve Fit'
if 'z' in best_fit_params:
    title += f' (z = {best_fit_params["z"]:.4f})'
plt.title(title, fontsize=14)

# Add legend
plt.legend(ncol=2, fontsize=10)
plt.grid(True, alpha=0.3)

# Save plot
plt.savefig('light_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Print best fit parameters
print("\nBest Fit Parameters:")
print("-" * 50)
for param, value in best_fit_params.items():
    print(f"{param}: {value:.6f}") 