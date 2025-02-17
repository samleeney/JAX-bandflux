import numpy as np
import matplotlib.pyplot as plt
from jax_supernovae.salt3 import initialize_model, get_flux
import jax.numpy as jnp

def plot_model_comparison(phase=0.0, x1=0.0, c=0.0):
    """
    Plot comparison between SALT3 and SALT3-NIR models at a specific phase.
    
    Parameters
    ----------
    phase : float
        Phase in days relative to peak brightness
    x1 : float
        SALT stretch parameter
    c : float
        SALT color parameter
    """
    # Set up wavelength grid
    wave = np.linspace(3000, 18000, 1000)
    
    # Plot settings
    plt.figure(figsize=(10, 6))
    
    # Plot SALT3-NIR
    initialize_model(model_type='salt3-nir')
    flux_nir = get_flux(jnp.array([phase]), jnp.array(wave), x1, c)[0]
    plt.plot(wave, flux_nir, label='SALT3-NIR', color='blue', alpha=0.7)
    
    # Plot SALT3
    initialize_model(model_type='salt3')
    flux_salt3 = get_flux(jnp.array([phase]), jnp.array(wave), x1, c)[0]
    plt.plot(wave, flux_salt3, label='SALT3', color='red', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f'SALT3 vs SALT3-NIR Model Comparison\nPhase={phase:.1f}, x1={x1:.1f}, c={c:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show wavelength ranges
    plt.axvspan(3000, 7000, color='gray', alpha=0.1, label='SALT3 range')
    plt.axvspan(7000, 18000, color='yellow', alpha=0.1, label='NIR extension')
    
    # Add text box with parameters
    plt.text(0.02, 0.98, 
             f'Parameters:\nPhase: {phase:.1f} days\nx1: {x1:.1f}\nc: {c:.2f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Plot for a standard SN Ia at peak brightness
    plot_model_comparison(phase=0.0, x1=0.0, c=0.0)
    print("Plot saved as 'model_comparison.png'") 