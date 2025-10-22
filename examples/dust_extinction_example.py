"""Example script demonstrating dust extinction in JAX-bandflux.

This script shows how to use the dust extinction laws with the SALT3 model
in JAX-bandflux. It creates a simple light curve with and without dust extinction
and plots the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sncosmo
from jax_supernovae import SALT3Source
from jax_supernovae.dust import ccm89_extinction, od94_extinction, f99_extinction

# Enable float64 precision
jax.config.update("jax_enable_x64", True)


def main():
    """Run the example script."""
    # Create SALT3 source for bandflux calculations
    source = SALT3Source()

    # Create a set of bandpasses (SDSS filters)
    sdss_bands = ['g', 'r', 'i', 'z']
    
    # Set up parameters for the SALT3 model
    base_params = {
        'z': 0.1,
        't0': 0.0,
        'x0': 1e-5,
        'x1': 0.0,
        'c': 0.0
    }
    
    # Create parameters with different dust laws
    params_no_dust = base_params.copy()
    
    # Create parameters for each dust law
    params_ccm89 = base_params.copy()
    params_ccm89.update({
        'dust_type': 0,  # 0 = CCM89
        'ebv': 0.1,
        'r_v': 3.1
    })
    
    params_od94 = base_params.copy()
    params_od94.update({
        'dust_type': 1,  # 1 = OD94
        'ebv': 0.1,
        'r_v': 3.1
    })
    
    params_f99 = base_params.copy()
    params_f99.update({
        'dust_type': 2,  # 2 = F99
        'ebv': 0.1,
        'r_v': 3.1
    })
    
    # Create a time grid
    phases = np.linspace(-10, 30, 41)

    # Calculate fluxes for each bandpass and parameter set
    fluxes_no_dust = np.zeros((len(phases), len(sdss_bands)))
    fluxes_ccm89 = np.zeros((len(phases), len(sdss_bands)))
    fluxes_od94 = np.zeros((len(phases), len(sdss_bands)))
    fluxes_f99 = np.zeros((len(phases), len(sdss_bands)))

    # For each dust scenario, calculate all fluxes at once
    for j, phase in enumerate(phases):
        for i, band_name in enumerate(sdss_bands):
            # No dust
            flux = source.bandflux(params_no_dust, band_name, phase)
            fluxes_no_dust[j, i] = float(flux)

            # CCM89 dust
            flux = source.bandflux(params_ccm89, band_name, phase)
            fluxes_ccm89[j, i] = float(flux)

            # OD94 dust
            flux = source.bandflux(params_od94, band_name, phase)
            fluxes_od94[j, i] = float(flux)

            # F99 dust
            flux = source.bandflux(params_f99, band_name, phase)
            fluxes_f99[j, i] = float(flux)
    
    # Convert fluxes to magnitudes
    zp = 25.0  # Arbitrary zero point
    mags_no_dust = -2.5 * np.log10(fluxes_no_dust) + zp
    mags_ccm89 = -2.5 * np.log10(fluxes_ccm89) + zp
    mags_od94 = -2.5 * np.log10(fluxes_od94) + zp
    mags_f99 = -2.5 * np.log10(fluxes_f99) + zp
    
    # Plot the light curves
    plt.figure(figsize=(12, 8))
    
    colors = ['g', 'r', 'i', 'z']
    markers = ['o', 's', '^', 'd']
    
    for i, (band_name, color, marker) in enumerate(zip(sdss_bands, colors, markers)):
        plt.subplot(2, 2, i+1)
        
        plt.plot(phases, mags_no_dust[:, i], 'k-', label='No dust')
        plt.plot(phases, mags_ccm89[:, i], 'r--', label='CCM89')
        plt.plot(phases, mags_od94[:, i], 'g-.', label='OD94')
        plt.plot(phases, mags_f99[:, i], 'b:', label='F99')
        
        plt.gca().invert_yaxis()  # Invert y-axis for magnitudes
        plt.xlabel('Phase (days)')
        plt.ylabel('Magnitude')
        plt.title(f'SDSS {band_name} band')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dust_extinction_example.png')
    plt.show()
    
    # Demonstrate the dust extinction laws directly
    demonstrate_dust_laws()
    
    print("Example completed. Light curves with different dust laws have been plotted.")


def demonstrate_dust_laws():
    """Demonstrate the dust extinction laws directly."""
    # Create a wavelength grid
    wave = np.linspace(2000, 10000, 1000)
    
    # Set dust parameters
    ebv = 0.1
    r_v = 3.1
    
    # Calculate extinction for each dust law
    ccm89_ext = ccm89_extinction(wave, ebv, r_v)
    od94_ext = od94_extinction(wave, ebv, r_v)
    f99_ext = f99_extinction(wave, ebv, r_v)
    
    # Calculate the corresponding sncosmo extinction for comparison
    ccm89_sncosmo = sncosmo.CCM89Dust()
    ccm89_sncosmo.parameters = [ebv, r_v]
    ccm89_sncosmo_flux = ccm89_sncosmo.propagate(wave, np.ones_like(wave))
    ccm89_sncosmo_ext = -2.5 * np.log10(ccm89_sncosmo_flux)
    
    od94_sncosmo = sncosmo.OD94Dust()
    od94_sncosmo.parameters = [ebv, r_v]
    od94_sncosmo_flux = od94_sncosmo.propagate(wave, np.ones_like(wave))
    od94_sncosmo_ext = -2.5 * np.log10(od94_sncosmo_flux)
    
    f99_sncosmo = sncosmo.F99Dust(r_v=r_v)
    f99_sncosmo.parameters = [ebv]
    f99_sncosmo_flux = f99_sncosmo.propagate(wave, np.ones_like(wave))
    f99_sncosmo_ext = -2.5 * np.log10(f99_sncosmo_flux)
    
    # Plot the extinction curves
    plt.figure(figsize=(12, 8))
    
    # Plot CCM89
    plt.subplot(2, 2, 1)
    plt.plot(wave, ccm89_ext, 'r-', label='JAX-bandflux')
    plt.plot(wave, ccm89_sncosmo_ext, 'k--', label='sncosmo')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Extinction (mag)')
    plt.title('CCM89 Dust Law')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot OD94
    plt.subplot(2, 2, 2)
    plt.plot(wave, od94_ext, 'g-', label='JAX-bandflux')
    plt.plot(wave, od94_sncosmo_ext, 'k--', label='sncosmo')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Extinction (mag)')
    plt.title('OD94 Dust Law')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot F99
    plt.subplot(2, 2, 3)
    plt.plot(wave, f99_ext, 'b-', label='JAX-bandflux')
    plt.plot(wave, f99_sncosmo_ext, 'k--', label='sncosmo')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Extinction (mag)')
    plt.title('F99 Dust Law')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot all laws together
    plt.subplot(2, 2, 4)
    plt.plot(wave, ccm89_ext, 'r-', label='CCM89')
    plt.plot(wave, od94_ext, 'g-', label='OD94')
    plt.plot(wave, f99_ext, 'b-', label='F99')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Extinction (mag)')
    plt.title('Comparison of Dust Laws')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dust_extinction_laws.png')
    plt.show()
    
    # Focus on the UV region where issues were occurring
    plt.figure(figsize=(12, 6))
    
    # Plot the UV region
    plt.subplot(1, 2, 1)
    plt.plot(wave, ccm89_ext, 'r-', label='CCM89')
    plt.plot(wave, od94_ext, 'g-', label='OD94')
    plt.plot(wave, f99_ext, 'b-', label='F99')
    plt.xlim(2000, 4000)  # Focus on UV region
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Extinction (mag)')
    plt.title('Dust Laws in UV Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot the ratio of JAX to sncosmo implementations
    plt.subplot(1, 2, 2)
    plt.plot(wave, ccm89_ext / ccm89_sncosmo_ext, 'r-', label='CCM89')
    plt.plot(wave, od94_ext / od94_sncosmo_ext, 'g-', label='OD94')
    plt.plot(wave, f99_ext / f99_sncosmo_ext, 'b-', label='F99')
    plt.axhline(y=1.0, color='k', linestyle='--')
    plt.xlim(2000, 4000)  # Focus on UV region
    plt.ylim(0.9, 1.1)    # Focus on ratio near 1.0
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('JAX / sncosmo Ratio')
    plt.title('Ratio of JAX to sncosmo Implementations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dust_extinction_uv_comparison.png')
    plt.show()


if __name__ == "__main__":
    main()