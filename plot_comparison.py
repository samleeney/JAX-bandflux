import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes
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

# Load chains for both runs
base_params = ['t0', 'log_x0', 'x1', 'c'] if fix_z else ['z', 't0', 'log_x0', 'x1', 'c']

# Load standard chains (without log_p)
standard_samples = read_chains('chains_standard/chains_standard', columns=base_params)

# Load anomaly chains (with log_p)
anomaly_params = base_params + ['log_p']
anomaly_samples = read_chains('chains_anomaly/chains_anomaly', columns=anomaly_params)

# Use the appropriate parameter names for plotting
param_names = base_params  # Only plot the common parameters between both chains

# Create overlaid corner plot
fig, axes = make_2d_axes(param_names, figsize=(10, 10), facecolor='w')
standard_samples.plot_2d(axes, alpha=0.7, label="Standard")
anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")

axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)
plt.savefig('corner_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a separate corner plot for log_p correlations in anomaly case
if 'log_p' in anomaly_samples.columns:
    log_p_params = base_params + ['log_p']
    fig, axes = make_2d_axes(log_p_params, figsize=(12, 12), facecolor='w')
    anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
    plt.suptitle('Anomaly Detection Corner Plot (including log_p)', fontsize=14)
    axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)
    plt.savefig('corner_anomaly_logp.png', dpi=300, bbox_inches='tight')
    plt.close()

# Now plot light curves
def get_model_curve(samples, percentile=50):
    """Get model curve for given percentile of parameters."""
    params = {}
    for param in param_names:
        if param != 'log_p':  # Skip logp as it's not needed for the model
            params[param] = float(np.percentile(samples[param], percentile))
    if 'log_x0' in params:
        params['x0'] = 10**params['log_x0']
        del params['log_x0']  # Remove log_x0 as we now have x0
    if fix_z:
        params['z'] = fixed_z[0]
    return params

# Create time grid for smooth model curves
t_min = np.min(times) - 5
t_max = np.max(times) + 5
t_grid = np.linspace(t_min, t_max, 100)

# Get unique bands
unique_bands = np.unique(band_indices)
n_bands = len(unique_bands)

# Set up the plot
plt.figure(figsize=(15, 10))

# Define colours for each band
colours = ['g', 'c', 'orange', 'r']  # g, c, o, r bands
markers = ['o', 's', 'D', '^']

# Plot data points for each band
for i, band_idx in enumerate(unique_bands):
    mask = band_indices == band_idx
    plt.errorbar(times[mask], fluxes[mask], yerr=fluxerrs[mask],
                fmt=markers[i], color=colours[i], label=f'Band {i} Data',
                markersize=8, alpha=0.6)

# Calculate and plot model curves for both standard and anomaly
for name, samples in [("Standard", standard_samples), ("Anomaly", anomaly_samples)]:
    params = get_model_curve(samples)
    linestyle = '--' if name == "Standard" else '-'
    
    for i in range(n_bands):
        # Create band indices array for this band
        band_idx_grid = np.full_like(t_grid, i, dtype=int)
        
        # Calculate model fluxes
        model_fluxes = optimized_salt3nir_multiband_flux(
            jnp.array(t_grid),
            bridges,
            params,
            zps=zps,
            zpsys='ab'
        )
        
        # Extract fluxes for this band
        band_fluxes = model_fluxes[:, i]
        
        # Plot model curve
        plt.plot(t_grid, band_fluxes, linestyle, color=colours[i], 
                label=f'Band {i} {name}', linewidth=2, alpha=0.8)

# Add labels and title
plt.xlabel('MJD', fontsize=12)
plt.ylabel('Flux', fontsize=12)
title = 'Light Curve Fit Comparison'
if 'z' in params:
    title += f' (z = {params["z"]:.4f})'
plt.title(title, fontsize=14)

# Add legend
plt.legend(ncol=2, fontsize=10)
plt.grid(True, alpha=0.3)

# Save plot
plt.savefig('light_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print parameter statistics for both
print("\nParameter Statistics Comparison:")
print("-" * 50)
for param in param_names:
    std_mean = standard_samples[param].mean()
    std_std = standard_samples[param].std()
    anom_mean = anomaly_samples[param].mean()
    anom_std = anomaly_samples[param].std()
    print(f"\n{param}:")
    print(f"  Standard: {std_mean:.6f} ± {std_std:.6f}")
    print(f"  Anomaly:  {anom_mean:.6f} ± {anom_std:.6f}")

# Print log_p statistics for anomaly case
if 'log_p' in anomaly_samples.columns:
    print("\nlog_p (Anomaly only):")
    log_p_mean = anomaly_samples['log_p'].mean()
    log_p_std = anomaly_samples['log_p'].std()
    print(f"  Mean: {log_p_mean:.6f} ± {log_p_std:.6f}")
    print(f"  Max: {anomaly_samples['log_p'].max():.6f}")
    print(f"  Min: {anomaly_samples['log_p'].min():.6f}") 