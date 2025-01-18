import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision

import numpy as np
import sncosmo
from jax_supernovae.salt2 import (
    salt2_m0, salt2_m1, salt2_colorlaw, salt2_flux,
    M0_data, M1_data, wave_grid, phase_grid
)

def test_m0_template():
    """Test that M0 template interpolation matches SNCosmo."""
    # Get SNCosmo's SALT2 source
    snc_source = sncosmo.get_source('salt2')
    
    # Test at grid points
    phase_grid_np = np.array(phase_grid)
    wave_grid_np = np.array(wave_grid)
    for i, phase in enumerate(phase_grid_np):
        for j, wave in enumerate(wave_grid_np):
            snc_m0 = snc_source._model['M0'](phase, wave)
            jax_m0 = M0_data[i, j]
            np.testing.assert_allclose(jax_m0, snc_m0, rtol=1e-10)
    
    # Test at interpolated points
    phases = np.array([-5.0, 0.0, 5.0, 10.0])
    waves = np.array([4000.0, 5000.0, 6000.0])
    for phase in phases:
        for wave in waves:
            snc_m0 = snc_source._model['M0'](phase, wave)
            jax_m0 = np.array(salt2_m0(phase, wave))
            np.testing.assert_allclose(jax_m0, snc_m0, rtol=1e-5)

def test_m1_template():
    """Test that M1 template interpolation matches SNCosmo."""
    # Get SNCosmo's SALT2 source
    snc_source = sncosmo.get_source('salt2')
    
    # Test at grid points
    phase_grid_np = np.array(phase_grid)
    wave_grid_np = np.array(wave_grid)
    for i, phase in enumerate(phase_grid_np):
        for j, wave in enumerate(wave_grid_np):
            snc_m1 = snc_source._model['M1'](phase, wave)
            jax_m1 = M1_data[i, j]
            np.testing.assert_allclose(jax_m1, snc_m1, rtol=1e-10)
    
    # Test at interpolated points
    phases = np.array([-5.0, 0.0, 5.0, 10.0])
    waves = np.array([4000.0, 5000.0, 6000.0])
    for phase in phases:
        for wave in waves:
            snc_m1 = snc_source._model['M1'](phase, wave)
            jax_m1 = np.array(salt2_m1(phase, wave))
            np.testing.assert_allclose(jax_m1, snc_m1, rtol=1e-5)

def test_colorlaw():
    """Test that color law calculation matches SNCosmo."""
    # Get SNCosmo's SALT2 source
    snc_source = sncosmo.get_source('salt2')
    
    # Test at various wavelengths
    waves = np.array([3000.0, 4000.0, 5000.0, 6000.0, 7000.0], dtype=np.float64)
    colorlaw_coeffs = np.array([-0.504294, 0.787691, -0.461715, 0.0815619])
    
    # Create a new color law instance
    colorlaw = sncosmo.salt2utils.SALT2ColorLaw([2800., 7000.], colorlaw_coeffs)
    
    # Test all wavelengths at once
    snc_cl = colorlaw(waves)
    jax_cl = np.array([salt2_colorlaw(w, colorlaw_coeffs) for w in waves])
    
    print("Color law comparison:")
    for w, s, j in zip(waves, snc_cl, jax_cl):
        print(f"Wave: {w}")
        print(f"SNCosmo CL: {s}")
        print(f"JAX CL: {j}")
        print(f"Ratio: {j/s}")
    
    np.testing.assert_allclose(jax_cl, snc_cl, rtol=1e-5)

def test_rest_frame_flux():
    """Test rest-frame flux calculation matches SNCosmo."""
    # Initialize models
    snc_source = sncosmo.get_source('salt2')
    
    # Set parameters
    x0, x1, c = 1e-5, 0.1, 0.2
    snc_source._parameters = np.array([x0, x1, c])
    params = {'x0': x0, 'x1': x1, 'c': c}
    
    # Create a new color law instance
    colorlaw_coeffs = np.array([-0.504294, 0.787691, -0.461715, 0.0815619])
    snc_source._colorlaw = sncosmo.salt2utils.SALT2ColorLaw([2800., 7000.], colorlaw_coeffs)
    
    # Test at various phases and wavelengths
    phases = np.array([-5.0, 0.0, 5.0, 10.0])
    waves = np.array([4000.0, 5000.0, 6000.0], dtype=np.float64)
    
    for phase in phases:
        # Test all wavelengths at once for this phase
        snc_flux = np.array([float(snc_source._flux(phase, np.array([w], dtype=np.float64))[0]) for w in waves])
        jax_flux = np.array([float(salt2_flux(phase, w, params)) for w in waves])
        
        print(f"Phase: {phase}")
        for w, s, j in zip(waves, snc_flux, jax_flux):
            print(f"Wave: {w}")
            print(f"SNCosmo flux: {s}")
            print(f"JAX flux: {j}")
            print(f"Ratio: {j/s}")
        
        np.testing.assert_allclose(jax_flux, snc_flux, rtol=1e-5)

def test_redshift_scaling():
    """Test redshift scaling matches SNCosmo."""
    # Initialize models
    snc_model = sncosmo.Model(source='salt2')
    
    # Set parameters
    params = {'t0': 55000.0, 'x0': 1e-5, 'x1': 0.1, 'c': 0.2, 'z': 0.5}
    snc_model.set(**params)
    
    # Create a new color law instance
    colorlaw_coeffs = np.array([-0.504294, 0.787691, -0.461715, 0.0815619])
    snc_model._source._colorlaw = sncosmo.salt2utils.SALT2ColorLaw([2800., 7000.], colorlaw_coeffs)
    
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
            m0_snc = snc_model._source._model['M0'](restphase, rw)
            m1_snc = snc_model._source._model['M1'](restphase, rw)
            # Create a writable array for the color law
            rw_array = np.array([rw], dtype=np.float64)
            cl_snc = snc_model._source._colorlaw(rw_array)[0]
            print(f"SNCosmo M0: {m0_snc}")
            print(f"SNCosmo M1: {m1_snc}")
            print(f"SNCosmo CL: {cl_snc}")
            
            # Get JAX components
            m0_jax = float(salt2_m0(restphase, rw))
            m1_jax = float(salt2_m1(restphase, rw))
            cl_jax = float(salt2_colorlaw(rw, colorlaw_coeffs))
            print(f"JAX M0: {m0_jax}")
            print(f"JAX M1: {m1_jax}")
            print(f"JAX CL: {cl_jax}")
            
            # Compare final fluxes
            snc_flux = float(snc_model._source._flux(restphase, rw_array)[0]) * a
            rest_flux = float(salt2_flux(restphase, rw, params))
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

def test_integration_grid():
    """Test integration grid matches SNCosmo."""
    # Get a bandpass
    band = sncosmo.get_bandpass('sdssg')
    
    # Get SNCosmo's integration grid
    snc_wave = np.array(band.wave)
    snc_trans = np.array(band.trans)
    
    # Print grid details
    print("SNCosmo grid:")
    print(f"Wave shape: {snc_wave.shape}, min: {snc_wave.min():.1f}, max: {snc_wave.max():.1f}")
    print(f"Trans shape: {snc_trans.shape}, min: {snc_trans.min():.4f}, max: {snc_trans.max():.4f}")
    
    # Calculate mean wavelength spacing
    snc_dwave = np.mean(np.diff(snc_wave))
    print(f"Mean wavelength spacing: {snc_dwave:.2f}") 