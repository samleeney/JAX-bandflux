import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes
from jax_supernovae.data import load_and_process_data
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
import os

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

# These settings will be overridden if the script is imported from ns_anomaly.py
if 'identifier' not in globals():
    # Set parameters
    fix_z = True  # Whether to fix the redshift
    fit_sigma = False  # Whether sigma is being fitted
    # Set identifier (should match the one used in ns_anomaly.py)
    identifier = "_v1"  # You can modify this or pass it as a command line argument
    # Load data
    sn_name = '21yrf'  # Define the supernova name

# Define output directories
output_dir = f'results/chains_{sn_name}{identifier}'

# Load data if not already loaded (will be already loaded if called from ns_anomaly.py)
if 'times' not in globals():
    times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data(sn_name, data_dir='jax_supernovae/data', fix_z=fix_z)

# Try to load weighted emax values
try:
    weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
    
    # Create a separate plot for weighted emax vs datapoint number
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(weighted_emax)), weighted_emax, 'k-', linewidth=2)
    plt.fill_between(np.arange(len(weighted_emax)), 0, weighted_emax, alpha=0.3)
    plt.xlabel('Data Point Number')
    plt.ylabel('Weighted Emax')
    plt.title('Weighted Emax by Data Point')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/weighted_emax.png', dpi=300, bbox_inches='tight')
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
    standard_samples = read_chains(f'{output_dir}/chains_standard', columns=base_params)
    have_standard = True
except FileNotFoundError:
    print("Warning: Standard chains not found - some plots will be incomplete")
    have_standard = False
    standard_samples = None

# Try to load anomaly chains with log_p parameter
try:
    anomaly_params = base_params + ['log_p']
    anomaly_samples = read_chains(f'{output_dir}/chains_anomaly', columns=anomaly_params)
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
    try:
        fig, axes = make_2d_axes(param_names, figsize=(10, 10), facecolor='w')
        
        # Plot standard samples if available
        if have_standard:
            try:
                standard_samples.plot_2d(axes, alpha=0.7, label="Standard")
            except Exception as e:
                print(f"Warning: Failed to plot standard samples in corner plot - {str(e)}")
        
        # Plot anomaly samples if available
        if have_anomaly:
            try:
                anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
            except Exception as e:
                print(f"Warning: Failed to plot anomaly samples in corner plot - {str(e)}")
        
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)
        plt.savefig(f'{output_dir}/corner_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to create corner comparison plot - {str(e)}")

# Create anomaly corner plot only if we have anomaly chains
if have_anomaly and 'log_p' in anomaly_samples.columns:
    try:
        log_p_params = base_params + ['log_p']
        fig, axes = make_2d_axes(log_p_params, figsize=(12, 12), facecolor='w')
        anomaly_samples.plot_2d(axes, alpha=0.7, label="Anomaly")
        plt.suptitle('Anomaly Detection Corner Plot (including log_p)', fontsize=14)
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)
        plt.savefig(f'{output_dir}/corner_anomaly_logp.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to create anomaly corner plot - {str(e)}")

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

try:
    # Create time grid for smooth model curves
    t_min = np.min(times) - 5
    t_max = np.max(times) + 5
    t_grid = np.linspace(t_min, t_max, 100)

    # Get unique bands
    unique_bands = np.unique(band_indices)
    n_bands = len(unique_bands)

    # Set up the plot with two subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.1)  # Reduced spacing between subplots

    # Main light curve plot
    ax1 = plt.subplot(gs[0])

    # Define colours for each band - use a colour map if more bands than colours
    default_colours = ['g', 'c', 'orange', 'r']
    default_markers = ['o', 's', 'D', '^']
    
    # Ensure we have enough colours and markers for all bands
    if n_bands > len(default_colours):
        colours = plt.cm.tab10(np.linspace(0, 1, n_bands))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '8'][:n_bands]
    else:
        colours = default_colours[:n_bands]
        markers = default_markers[:n_bands]

    # Load weighted emax values first to identify anomalous points
    try:
        weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
        plotting_threshold = 0.2
        
        # Count total anomalous points
        total_anomalous_points = 0
        
        # Print diagnostic info
        print(f"Number of weighted_emax values: {len(weighted_emax)}")
        print(f"Number of points below threshold: {np.sum(weighted_emax < plotting_threshold)}")
        
        # Create time points that match the actual data points
        all_times = np.sort(np.unique(times))
        if len(all_times) != len(weighted_emax):
            print(f"Warning: Number of unique time points ({len(all_times)}) "
                  f"doesn't match number of weighted_emax values ({len(weighted_emax)})")
        
        # Create time points for the emax plot - using actual data time points
        emax_times = all_times
    except FileNotFoundError:
        print("Warning: Weighted emax file not found")
        weighted_emax = None

    # Plot data points for each band
    try:
        for i, band_idx in enumerate(unique_bands):
            mask = band_indices == band_idx
            band_times = times[mask]
            band_fluxes = fluxes[mask]
            band_errors = fluxerrs[mask]

            if weighted_emax is not None:
                # Map each data point to its index in the sorted unique times
                time_indices = np.searchsorted(all_times, band_times)
                # Ensure indices are within bounds
                time_indices = np.clip(time_indices, 0, len(weighted_emax) - 1)
                # Get the emax value for each point
                point_emax = weighted_emax[time_indices]
                
                # Print diagnostic info for this band
                print(f"\nBand {i}:")
                print(f"Number of points: {len(band_times)}")
                print(f"Number of points below threshold: {np.sum(point_emax < plotting_threshold)}")
                
                # Determine which points are anomalous
                normal_mask = point_emax >= plotting_threshold
                anomaly_mask = point_emax < plotting_threshold
                
                # Update total count
                total_anomalous_points += np.sum(anomaly_mask)

                # Plot normal points
                if np.any(normal_mask):
                    ax1.errorbar(band_times[normal_mask], band_fluxes[normal_mask], 
                               yerr=band_errors[normal_mask],
                               fmt=markers[i], color=colours[i], 
                               label=f'Band {i} Data',
                               markersize=8, alpha=0.6)
                
                # Plot anomalous points with star markers
                if np.any(anomaly_mask):
                    label = f'Band {i} Anomalous' if np.any(normal_mask) else f'Band {i} Data'
                    ax1.errorbar(band_times[anomaly_mask], band_fluxes[anomaly_mask], 
                               yerr=band_errors[anomaly_mask],
                               fmt='*', color=colours[i], 
                               label=label,
                               markersize=15, alpha=0.8)
                    
                    # Print times of anomalous points
                    print(f"Anomalous point times: {band_times[anomaly_mask]}")
                    print(f"Corresponding emax values: {point_emax[anomaly_mask]}")
            else:
                # Plot all points normally if no weighted_emax available
                ax1.errorbar(band_times, band_fluxes, yerr=band_errors,
                           fmt=markers[i], color=colours[i], label=f'Band {i} Data',
                           markersize=8, alpha=0.6)
    except Exception as e:
        print(f"Warning: Failed to plot some data points - {str(e)}")

    # Calculate and plot model curves for both standard and anomaly if available
    if have_standard or have_anomaly:
        for name, samples, has_samples in [
            ("Standard", standard_samples, have_standard), 
            ("Anomaly", anomaly_samples, have_anomaly)
        ]:
            if has_samples:
                try:
                    params = get_model_curve(samples)
                    linestyle = '--' if name == "Standard" else '-'
                    
                    for i in range(n_bands):
                        try:
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
                        except Exception as e:
                            print(f"Warning: Failed to plot {name} model curve for band {i} - {str(e)}")
                            continue
                except Exception as e:
                    print(f"Warning: Failed to plot {name} model curves - {str(e)}")
                    continue

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
        weighted_emax = np.loadtxt(f'{output_dir}/chains_anomaly_weighted_emax.txt')
        ax2 = plt.subplot(gs[1])
        
        # Use actual data time points for the emax plot
        ax2.plot(all_times, weighted_emax, 'k-', linewidth=2)
        ax2.fill_between(all_times, 0, weighted_emax, alpha=0.3, color='gray')
        ax2.set_xlabel('MJD', fontsize=12)
        ax2.set_ylabel('Emax', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add horizontal line at threshold
        ax2.axhline(y=plotting_threshold, color='r', linestyle='--', alpha=0.5, 
                   label=f'Plotting threshold ({plotting_threshold}) - {total_anomalous_points} points below')
        ax2.legend()

        # Ensure x-axis limits match between plots
        xlim = ax1.get_xlim()
        ax2.set_xlim(xlim)

        # Remove x-axis labels from top plot
        ax1.set_xlabel('')
    except FileNotFoundError:
        print("Warning: Weighted emax file not found - skipping emax subplot")
        plt.tight_layout()  # Adjust layout even without the subplot
    except Exception as e:
        print(f"Warning: Failed to create emax subplot - {str(e)}")
        plt.tight_layout()  # Adjust layout even without the subplot

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/light_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Warning: Failed to create light curve comparison plot - {str(e)}")
    plt.close('all')  # Ensure all plots are closed even if we fail

# Save parameter statistics to a text file if chains are available
if have_standard or have_anomaly:
    stats_text = ["Parameter Statistics Comparison:", "-" * 50]
    for param in param_names:
        if have_standard:
            std_mean = standard_samples[param].mean()
            std_std = standard_samples[param].std()
            stats_text.append(f"\n{param}:")
            stats_text.append(f"  Standard: {std_mean:.6f} ± {std_std:.6f}")
        if have_anomaly:
            anom_mean = anomaly_samples[param].mean()
            anom_std = anomaly_samples[param].std()
            stats_text.append(f"  Anomaly:  {anom_mean:.6f} ± {anom_std:.6f}")

    # Add log_p statistics for anomaly case if available
    if have_anomaly and 'log_p' in anomaly_samples.columns:
        stats_text.extend([
            "\nlog_p (Anomaly only):",
            f"  Mean: {anomaly_samples['log_p'].mean():.6f} ± {anomaly_samples['log_p'].std():.6f}",
            f"  Max: {anomaly_samples['log_p'].max():.6f}",
            f"  Min: {anomaly_samples['log_p'].min():.6f}"
        ])

    # Save statistics to both directories if they exist
    stats_text = '\n'.join(stats_text)
    if have_standard:
        with open(f'{output_dir}/parameter_statistics.txt', 'w') as f:
            f.write(stats_text)
    if have_anomaly:
        with open(f'{output_dir}/parameter_statistics.txt', 'w') as f:
            f.write(stats_text) 