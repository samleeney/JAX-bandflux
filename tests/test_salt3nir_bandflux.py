import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision

import numpy as np
import sncosmo
from jax_supernovae.models import Model
from jax_supernovae.salt3nir import salt3nir_flux, salt3nir_m0, salt3nir_m1, salt3nir_colorlaw
from jax_supernovae.salt3nir import wave_grid, phase_grid

def test_salt3nir_bandflux():
    """Test that SALT3-NIR bandflux matches between sncosmo and JAX implementations."""
    
    # Create sncosmo model
    snc_model = sncosmo.Model(source='salt3-nir')
    params = {'z': 0.1, 't0': 0.0, 'x0': 1e-5, 'x1': 0.1, 'c': 0.1}
    snc_model.set(**params)
    
    # Create JAX model
    jax_model = Model()
    jax_model.wave = wave_grid
    jax_model.flux = lambda t, w: salt3nir_flux(t, w, jax_model.parameters)
    jax_model.parameters = params
    
    # Test bands and times
    bands = ['sdssg', 'sdssr', 'sdssi']  # Common SDSS bands
    times = np.linspace(-10, 40, 20)  # Cover typical SN Ia evolution
    
    for band in bands:
        print(f"\nTesting band: {band}")
        
        # Get fluxes from both implementations
        snc_flux = snc_model.bandflux(band, times)
        jax_flux = jax_model.bandflux(band, times)
        
        # Convert to numpy for analysis
        jax_flux = np.array(jax_flux)
        
        # Calculate differences
        abs_diff = np.abs(jax_flux - snc_flux)
        rel_diff = np.where(snc_flux != 0, abs_diff / np.abs(snc_flux), 0)
        
        print(f"Mean flux: {np.mean(snc_flux):.2e}")
        print(f"Mean absolute difference: {np.mean(abs_diff):.2e}")
        print(f"Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"Mean relative difference: {np.mean(rel_diff):.2%}")
        print(f"Max relative difference: {np.max(rel_diff):.2%}")
        
        # Assert that relative differences are small where flux is significant
        significant_flux = np.abs(snc_flux) > 1e-10
        if np.any(significant_flux):
            rel_diff_significant = rel_diff[significant_flux]
            np.testing.assert_allclose(
                jax_flux[significant_flux], 
                snc_flux[significant_flux],
                rtol=0.01,  # Allow 1% relative difference
                err_msg=f"Significant flux mismatch in {band} band"
            )

def test_salt3nir_components():
    """Test individual components of SALT3-NIR model."""
    
    # Create models
    snc_model = sncosmo.Model(source='salt3-nir')
    params = {'z': 0.0, 't0': 0.0, 'x0': 1.0, 'x1': 0.0, 'c': 0.0}
    snc_model.set(**params)
    
    # Test at specific time and wavelength
    t = 0.0
    wave = 5000.0
    
    # Get M0 component
    snc_m0 = snc_model._source._model['M0'](np.array([t]), np.array([wave]))[0][0]
    jax_m0 = salt3nir_m0(t, wave)
    
    # Get M1 component
    snc_m1 = snc_model._source._model['M1'](np.array([t]), np.array([wave]))[0][0]
    jax_m1 = salt3nir_m1(t, wave)
    
    # Get color law
    snc_cl = snc_model._source._colorlaw(np.array([wave], dtype=np.float64))[0]
    jax_cl = salt3nir_colorlaw(wave)
    
    print("\nComponent comparison at t=0, wave=5000:")
    print(f"M0 - SNCosmo: {snc_m0:.3e}, JAX: {float(jax_m0):.3e}")
    print(f"M1 - SNCosmo: {snc_m1:.3e}, JAX: {float(jax_m1):.3e}")
    print(f"CL - SNCosmo: {snc_cl:.3f}, JAX: {float(jax_cl):.3f}")
    
    # Test relative differences
    np.testing.assert_allclose(jax_m0, snc_m0, rtol=1e-4,
                              err_msg="M0 component mismatch")
    np.testing.assert_allclose(jax_m1, snc_m1, rtol=1e-4,
                              err_msg="M1 component mismatch")
    np.testing.assert_allclose(jax_cl, snc_cl, rtol=1e-4,
                              err_msg="Color law mismatch")

def test_salt3nir_redshift_scaling():
    """Test redshift scaling matches SNCosmo."""
    # Initialize models
    snc_model = sncosmo.Model(source='salt3-nir')
    
    # Set parameters
    params = {'t0': 55000.0, 'x0': 1e-5, 'x1': 0.1, 'c': 0.2, 'z': 0.5}
    snc_model.set(**params)
    
    # Test at various observer-frame times and wavelengths
    times = np.array([54990.0, 55000.0, 55010.0])
    waves = np.array([4000.0, 5000.0, 6000.0], dtype=np.float64)
    
    # Arrays to store all flux values for error analysis
    all_snc_fluxes = []
    all_jax_fluxes = []
    
    for time in times:
        # Calculate rest-frame quantities
        z = params['z']
        t0 = params['t0']
        a = 1.0 / (1.0 + z)
        restphase = (time - t0) * a
        restwaves = waves * a
        
        # Get SNCosmo flux components
        print(f"\nTime: {time}")
        print(f"Rest phase: {restphase}")
        for w, rw in zip(waves, restwaves):
            print(f"\nObserver wave: {w}, Rest wave: {rw}")
            
            # Get SNCosmo components
            m0_snc = snc_model._source._model['M0'](np.array([restphase]), np.array([rw]))[0][0]
            m1_snc = snc_model._source._model['M1'](np.array([restphase]), np.array([rw]))[0][0]
            cl_snc = snc_model._source._colorlaw(np.array([rw], dtype=np.float64))[0]
            print(f"SNCosmo M0: {m0_snc}")
            print(f"SNCosmo M1: {m1_snc}")
            print(f"SNCosmo CL: {cl_snc}")
            
            # Get JAX components
            m0_jax = float(salt3nir_m0(restphase, rw))
            m1_jax = float(salt3nir_m1(restphase, rw))
            cl_jax = float(salt3nir_colorlaw(rw))
            print(f"JAX M0: {m0_jax}")
            print(f"JAX M1: {m1_jax}")
            print(f"JAX CL: {cl_jax}")
            
            # Compare final fluxes
            snc_flux = float(snc_model._source._flux(np.array([restphase]), np.array([rw]))[0]) * a
            rest_flux = float(salt3nir_flux(restphase, rw, params))
            jax_flux = rest_flux * a
            
            print(f"SNCosmo flux: {snc_flux}")
            print(f"JAX flux: {jax_flux}")
            print(f"Ratio: {jax_flux/snc_flux}")
            
            all_snc_fluxes.append(snc_flux)
            all_jax_fluxes.append(jax_flux)
    
    # Convert to numpy arrays for analysis
    all_snc_fluxes = np.array(all_snc_fluxes)
    all_jax_fluxes = np.array(all_jax_fluxes)
    
    # Calculate error statistics
    abs_diff = np.abs(all_jax_fluxes - all_snc_fluxes)
    rel_diff = abs_diff / all_snc_fluxes
    
    print("\nError Statistics:")
    print(f"Mean absolute error: {np.mean(abs_diff):.2e}")
    print(f"Max absolute error: {np.max(abs_diff):.2e}")
    print(f"Mean relative error: {np.mean(rel_diff):.2%}")
    print(f"Max relative error: {np.max(rel_diff):.2%}")
    print(f"RMS relative error: {np.sqrt(np.mean(rel_diff**2)):.2%}")
    
    # Allow up to 1% difference due to interpolation differences
    np.testing.assert_allclose(all_jax_fluxes, all_snc_fluxes, rtol=1e-2) 