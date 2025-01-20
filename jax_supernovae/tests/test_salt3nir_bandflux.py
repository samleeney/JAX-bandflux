import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision

import numpy as np
import pytest
import sncosmo
from jax_supernovae.core import Bandpass, HC_ERG_AA, MODEL_BANDFLUX_SPACING
from jax_supernovae.salt3nir import salt3nir_bandflux, salt3nir_m0, salt3nir_m1, salt3nir_colorlaw
from jax_supernovae.salt3nir import integration_grid

def test_bandflux_matches_sncosmo():
    """Test that bandflux matches SNCosmo implementation."""
    # Create SNCosmo model
    snc_model = sncosmo.Model(source='salt3-nir')
    params = {'z': 0.1, 't0': 0.0, 'x0': 1e-5, 'x1': 0.1, 'c': 0.2}
    snc_model.update(params)
    
    # Test at various phases and bands
    phases = np.linspace(-10, 40, 10)
    bands = ['sdssg', 'sdssr', 'sdssi', 'sdssz']
    
    for phase in phases:
        for band_name in bands:
            print(f"\nTesting phase={phase}, band={band_name}")
            
            # Get bandpass
            snc_bandpass = sncosmo.get_bandpass(band_name)
            print(f"Bandpass wavelength range: [{snc_bandpass.wave[0]}, {snc_bandpass.wave[-1]}]")
            print(f"Bandpass transmission range: [{snc_bandpass.trans.min()}, {snc_bandpass.trans.max()}]")
            
            # Convert to JAX-compatible bandpass
            bandpass = Bandpass(snc_bandpass.wave, snc_bandpass.trans)
            
            # Get integration grid (both SNCosmo and JAX)
            wave, dwave = integration_grid(bandpass.minwave(), bandpass.maxwave(), MODEL_BANDFLUX_SPACING)
            print(f"\nIntegration grid:")
            print(f"Grid spacing (dwave): {dwave}")
            print(f"Number of points: {len(wave)}")
            print(f"Wave range: [{wave[0]}, {wave[-1]}]")
            
            # Get transmission on integration grid
            trans = bandpass(wave)
            print(f"Resampled transmission range: [{trans.min()}, {trans.max()}]")
            
            # Calculate rest-frame quantities
            z = params['z']
            t0 = params['t0']
            a = 1.0 / (1.0 + z)
            restphase = (phase - t0) * a
            restwave = wave * a
            print(f"\nRest frame quantities:")
            print(f"Scale factor (a): {a}")
            print(f"Rest phase: {restphase}")
            print(f"Rest wave range: [{restwave[0]}, {restwave[-1]}]")
            
            # Get model components
            m0 = salt3nir_m0(restphase, restwave)
            m1 = salt3nir_m1(restphase, restwave)
            cl = salt3nir_colorlaw(restwave)
            
            # Print component ranges and some sample values
            print(f"\nModel components:")
            print(f"M0 range: [{m0.min()}, {m0.max()}]")
            print(f"M1 range: [{m1.min()}, {m1.max()}]")
            print(f"Color law range: [{cl.min()}, {cl.max()}]")
            
            # Calculate flux components
            x0 = params['x0']
            x1 = params['x1']
            c = params['c']
            rest_flux = x0 * (m0 + x1 * m1) * 10**(-0.4 * cl * c)
            print(f"\nFlux components:")
            print(f"Rest flux range: [{rest_flux.min()}, {rest_flux.max()}]")
            
            # Calculate integrand components
            integrand = wave * trans * rest_flux
            print(f"\nIntegration:")
            print(f"Integrand range: [{integrand.min()}, {integrand.max()}]")
            print(f"Sum of integrand: {integrand.sum()}")
            print(f"dwave * sum / HC_ERG_AA: {integrand.sum() * dwave / HC_ERG_AA}")
            
            # Get fluxes
            snc_flux = snc_model.bandflux(band_name, phase)
            jax_flux = salt3nir_bandflux(phase, bandpass, params)
            
            print(f"\nFinal results:")
            print(f"SNCosmo flux: {snc_flux:.6e}")
            print(f"JAX flux: {float(jax_flux):.6e}")
            print(f"Ratio (JAX/SNCosmo): {float(jax_flux/snc_flux):.6f}")
            
            # Compare results
            np.testing.assert_allclose(
                snc_flux, jax_flux, rtol=1e-5,
                err_msg=f"Bandflux mismatch at phase {phase}, band {band_name}\n"
                       f"SNCosmo: {snc_flux:.6e}\n"
                       f"JAX:     {float(jax_flux):.6e}\n"
                       f"Ratio:   {float(jax_flux/snc_flux):.6f}"
            ) 