import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes
from jax_supernovae.data import load_and_process_data
from jax_supernovae.salt3 import optimized_salt3_multiband_flux

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# Set parameters
fix_z = True  # Whether to fix the redshift
fit_sigma = False  # Whether sigma is being fitted

# Load data
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19dwz', data_dir='data', fix_z=fix_z)

# Try to load weighted emax values
try:
    weighted_emax = np.loadtxt('chains_anomaly/chains_anomaly_weighted_emax.txt')
    
    # Create a separate plot for weighted emax vs datapoint number
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(weighted_emax)), weighted_emax, 'k-', linewidth=2)
    plt.fill_between(np.arange(len(weighted_emax)), 0, weighted_emax, alpha=0.3)
    plt.xlabel('Data Point Number')
    plt.ylabel('Weighted Emax')
    plt.title('Weighted Emax by Data Point')
    plt.grid(True, alpha=0.3)
    plt.savefig('weighted_emax.png', dpi=300, bbox_inches='tight')
    plt.close()
except FileNotFoundError:
    print("Warning: Weighted emax file not found - skipping initial emax plot")

# Define parameter names based on settings
if fix_z:
    base_params = ['t0', 'log_x0', 'x1', 'c']
else:
    base_params = ['z', 't0', 'log_x0', 'x1', 'c']

if fit_sigma:
    base_params.append('sigma')

# Try to load standard chains
try:
    standard_samples = read_chains('chains_standard/chains_standard', columns=base_params)
    have_standard = True
except FileNotFoundError:
    print("Warning: Standard chains not found - some plots will be incomplete")
    have_standard = False
    standard_samples = None

# Try to load anomaly chains with log_p parameter
try:
    anomaly_params = base_params + ['log_p']
    anomaly_samples = read_chains('chains_anomaly/chains_anomaly', columns=anomaly_params)
    have_anomaly = True
except FileNotFoundError:
    print("Warning: Anomaly chains not found - some plots will be incomplete")
    have_anomaly = False
    anomaly_samples = None

# Use the appropriate parameter names for plotting
param_names = base_params  # Only plot the common parameters between both chains

# Only create corner plot if we have at least one set of chains
if have_standard or have_anomaly:
    # Create overlaid corner plot
    fig, axes = make_2d_axes(param_names, figsize=(10, 10), facecolor='w')
    if have_standard:
        standard_samples.plot_2d(axes, alpha=0.7, label="Standard")
    if have_anomaly:
        anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
    
    axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)
    plt.savefig('corner_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create anomaly corner plot only if we have anomaly chains
if have_anomaly and 'log_p' in anomaly_samples.columns:
    log_p_params = base_params + ['log_p']
    fig, axes = make_2d_axes(log_p_params, figsize=(12, 12), facecolor='w')
    anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
    plt.suptitle('Anomaly Detection Corner Plot (including log_p)', fontsize=14)
    axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)
    plt.savefig('corner_anomaly_logp.png', dpi=300, bbox_inches='tight')
    plt.close()

# Now plot light curves with weighted emax subplot
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
    if not fit_sigma:
        params['sigma'] = 1.0  # Add default sigma if not fitted
    return params

# Create time grid for smooth model curves
t_min = np.min(times) - 5
t_max = np.max(times) + 5
t_grid = np.linspace(t_min, t_max, 100)

# Get unique bands
unique_bands = np.unique(band_indices)
n_bands = len(unique_bands)

# Set up the plot with two subplots
fig = plt.figure(figsize=(15, 12))
gs = plt.GridSpec(2, 1, height_ratios=[4, 1])  # 2 rows with 4:1 ratio

# Main light curve plot
ax1 = plt.subplot(gs[0])

# Define colours for each band
colours = ['g', 'c', 'orange', 'r']  # g, c, o, r bands
markers = ['o', 's', 'D', '^']

# Plot data points for each band
for i, band_idx in enumerate(unique_bands):
    mask = band_indices == band_idx
    ax1.errorbar(times[mask], fluxes[mask], yerr=fluxerrs[mask],
                fmt=markers[i], color=colours[i], label=f'Band {i} Data',
                markersize=8, alpha=0.6)

# Calculate and plot model curves for both standard and anomaly if available
if have_standard or have_anomaly:
    for name, samples, has_samples in [
        ("Standard", standard_samples, have_standard), 
        ("Anomaly", anomaly_samples, have_anomaly)
    ]:
        if has_samples:
            params = get_model_curve(samples)
            linestyle = '--' if name == "Standard" else '-'
            
            for i in range(n_bands):
                # Create band indices array for this band
                band_idx_grid = np.full_like(t_grid, i, dtype=int)
                
                # Calculate model fluxes
                model_fluxes = optimized_salt3_multiband_flux(
                    jnp.array(t_grid),
                    bridges,
                    params,
                    zps=zps,
                    zpsys='ab'
                )
                
                # Extract fluxes for this band
                band_fluxes = model_fluxes[:, i]
                
                # Plot model curve
                ax1.plot(t_grid, band_fluxes, linestyle, color=colours[i], 
                        label=f'Band {i} {name}', linewidth=2, alpha=0.8)

# Add labels and title to main plot
ax1.set_xlabel('MJD', fontsize=12)
ax1.set_ylabel('Flux', fontsize=12)
title = 'Light Curve Fit Comparison'
if fix_z:
    title += f' (z = {fixed_z[0]:.4f})'
ax1.set_title(title, fontsize=14)

# Add legend to main plot
ax1.legend(ncol=2, fontsize=10)
ax1.grid(True, alpha=0.3)

# Try to load weighted emax values and create subplot
try:
    weighted_emax = np.loadtxt('chains_anomaly/chains_anomaly_weighted_emax.txt')
    ax2 = plt.subplot(gs[1])
    ax2.plot(times, weighted_emax, 'k-', linewidth=2)
    ax2.fill_between(times, 0, weighted_emax, alpha=0.3, color='gray')
    ax2.set_xlabel('MJD', fontsize=12)
    ax2.set_ylabel('Emax', fontsize=12)
    ax2.grid(True, alpha=0.3)
except FileNotFoundError:
    print("Warning: Weighted emax file not found - skipping emax subplot")

# Adjust layout and save
plt.tight_layout()
plt.savefig('light_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print parameter statistics if chains are available
if have_standard or have_anomaly:
    print("\nParameter Statistics Comparison:")
    print("-" * 50)
    for param in param_names:
        if have_standard:
            std_mean = standard_samples[param].mean()
            std_std = standard_samples[param].std()
            print(f"\n{param}:")
            print(f"  Standard: {std_mean:.6f} ± {std_std:.6f}")
        if have_anomaly:
            anom_mean = anomaly_samples[param].mean()
            anom_std = anomaly_samples[param].std()
            print(f"  Anomaly:  {anom_mean:.6f} ± {anom_std:.6f}")

# Print log_p statistics for anomaly case if available
if have_anomaly and 'log_p' in anomaly_samples.columns:
    print("\nlog_p (Anomaly only):")
    log_p_mean = anomaly_samples['log_p'].mean()
    log_p_std = anomaly_samples['log_p'].std()
    print(f"  Mean: {log_p_mean:.6f} ± {log_p_std:.6f}")
    print(f"  Max: {anomaly_samples['log_p'].max():.6f}")
    print(f"  Min: {anomaly_samples['log_p'].min():.6f}") 