import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # Enable float64 precision

import numpy as np
import sncosmo
import time
from jax_supernovae.core import Bandpass
from jax_supernovae.salt3nir import (salt3nir_bandflux, salt3nir_m0, 
                                    salt3nir_m1, salt3nir_colorlaw,
                                    salt3nir_multiband_flux)
from jax_supernovae.bandpasses import register_bandpass

def time_function(func, args, n_trials=100, warmup=True, name="Function"):
    """Time a function over multiple trials."""
    if warmup:
        # Warmup call for JIT compilation
        _ = func(*args)
    
    # Time the function
    start_time = time.time()
    for _ in range(n_trials):
        _ = func(*args)
    avg_time = (time.time() - start_time) / n_trials
    
    print(f"{name}: {avg_time*1000:8.3f} ms per call")
    return avg_time

def test_function_runtimes():
    """Test the runtime of individual functions."""
    # Create test data
    phase = jnp.linspace(-10, 50, 100)
    wave = jnp.linspace(3000, 9000, 1000)
    
    # Create test bandpass
    bandpass = Bandpass(wave, jnp.ones_like(wave))
    bandpasses = tuple([bandpass] * 5)
    
    # Test parameters
    params = {
        'z': 0.4,
        't0': 55098.,
        'x0': 1e-5,
        'x1': 0.,
        'c': 0.
    }
    
    print("\nTesting individual function runtimes...")
    print("-" * 60)
    print("Input sizes:")
    print(f"Phase points: {len(phase)}")
    print(f"Wavelength points: {len(wave)}")
    print(f"Number of bandpasses: {len(bandpasses)}")
    print("-" * 60)
    
    # Test M0 function
    m0_time = time_function(
        salt3nir_m0,
        (phase[:, None], wave[None, :]),
        name="salt3nir_m0"
    )
    
    # Test M1 function
    m1_time = time_function(
        salt3nir_m1,
        (phase[:, None], wave[None, :]),
        name="salt3nir_m1"
    )
    
    # Test colorlaw function
    cl_time = time_function(
        salt3nir_colorlaw,
        (wave,),
        name="salt3nir_colorlaw"
    )
    
    # Test bandflux function
    bf_time = time_function(
        salt3nir_bandflux,
        (phase, bandpass, params),
        name="salt3nir_bandflux"
    )
    
    # Test multiband flux function
    mbf_time = time_function(
        salt3nir_multiband_flux,
        (phase, bandpasses, params),
        name="salt3nir_multiband_flux"
    )
    
    print("-" * 60)
    print("Time breakdown:")
    print(f"M0 component:      {m0_time*1000:8.3f} ms ({m0_time/mbf_time*100:5.1f}% of multiband)")
    print(f"M1 component:      {m1_time*1000:8.3f} ms ({m1_time/mbf_time*100:5.1f}% of multiband)")
    print(f"Color law:         {cl_time*1000:8.3f} ms ({cl_time/mbf_time*100:5.1f}% of multiband)")
    print(f"Single bandflux:   {bf_time*1000:8.3f} ms ({bf_time/mbf_time*100:5.1f}% of multiband)")
    print(f"Multiband flux:    {mbf_time*1000:8.3f} ms (100.0% reference)")
    print("-" * 60)

def test_objective_function_runtime():
    """Compare runtime of JAX and SNCosmo objective functions."""
    # Create test data (similar to sncosmo example)
    data = sncosmo.load_example_data()
    
    # Create SNCosmo model
    snc_model = sncosmo.Model(source='salt2')
    
    # Create JAX bandpasses
    band_dict = {}
    for band_name in np.unique(data['band']):
        snc_band = sncosmo.get_bandpass(band_name)
        band_dict[band_name] = Bandpass(snc_band.wave, snc_band.trans)
        register_bandpass(band_name, band_dict[band_name], force=True)
    
    # Define objective function for sncosmo (from example)
    def objective_sncosmo(parameters):
        snc_model.parameters[:] = parameters
        model_flux = snc_model.bandflux(data['band'], data['time'],
                                      zp=data['zp'], zpsys=data['zpsys'])
        return np.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)
    
    # Define objective function for JAX implementation (from main.py)
    @jax.jit
    def objective_jax(parameters):
        params = {
            'z': parameters[0],
            't0': parameters[1],
            'x0': parameters[2],
            'x1': parameters[3],
            'c': parameters[4]
        }
        
        # Convert data to JAX arrays once
        times = jnp.array(data['time'])
        fluxes = jnp.array(data['flux'])
        fluxerrs = jnp.array(data['fluxerr'])
        zps = jnp.array(data['zp'])
        
        # Create list of unique bandpasses and map band names to indices
        unique_bands = []
        band_list = []
        for band in data['band']:
            if band not in unique_bands:
                unique_bands.append(band)
            band_list.append(unique_bands.index(band))
        band_indices = jnp.array(band_list)
        bandpasses = [band_dict[band] for band in unique_bands]
        
        # Calculate model flux for each observation
        model_fluxes = jnp.zeros_like(fluxes)
        for i in range(len(times)):
            flux = salt3nir_bandflux(jnp.array([times[i]]), bandpasses[band_list[i]], 
                                   params, zp=zps[i], zpsys='ab')[0]
            model_fluxes = model_fluxes.at[i].set(flux)
        
        # Calculate chi-squared using JAX operations
        chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
        
        return chi2
    
    # Test parameters (same as example)
    test_params = [0.4, 55098., 1e-5, 0., 0.]  # z, t0, x0, x1, c
    
    # Warm up JAX by running multiple times
    print("\nWarming up JAX (compilation)...")
    for _ in range(10):
        _ = objective_jax(test_params)
    
    # Time SNCosmo version
    n_trials = 100
    n_batches = 5
    print(f"\nTiming objective functions over {n_batches} batches of {n_trials} trials...")
    print("-" * 60)
    
    # Time SNCosmo
    snc_times = []
    for batch in range(n_batches):
        start_time = time.time()
        for _ in range(n_trials):
            _ = objective_sncosmo(test_params)
        batch_time = (time.time() - start_time) / n_trials
        snc_times.append(batch_time)
        print(f"SNCosmo batch {batch+1}: {batch_time*1000:8.3f} ms per call")
    
    # Time JAX
    jax_times = []
    for batch in range(n_batches):
        start_time = time.time()
        for _ in range(n_trials):
            _ = objective_jax(test_params)
        batch_time = (time.time() - start_time) / n_trials
        jax_times.append(batch_time)
        print(f"JAX batch {batch+1}:    {batch_time*1000:8.3f} ms per call")
    
    # Print average results
    snc_avg = np.mean(snc_times)
    jax_avg = np.mean(jax_times)
    print("\nAverage time per call:")
    print(f"SNCosmo: {snc_avg*1000:8.3f} ms (± {np.std(snc_times)*1000:6.3f} ms)")
    print(f"JAX:     {jax_avg*1000:8.3f} ms (± {np.std(jax_times)*1000:6.3f} ms)")
    print(f"Speedup: {snc_avg/jax_avg:8.2f}x")
    print("-" * 60)
    
    # Compare output values
    snc_val = objective_sncosmo(test_params)
    jax_val = objective_jax(test_params)
    print(f"\nObjective function values:")
    print(f"SNCosmo: {snc_val:10.4f}")
    print(f"JAX:     {jax_val:10.4f}")
    print(f"Ratio:   {jax_val/snc_val:10.4f}")
    
    # Assert that the values are close
    np.testing.assert_allclose(jax_val, snc_val, rtol=1e-2,
                             err_msg="Objective function values do not match")

if __name__ == "__main__":
    test_function_runtimes()
    test_objective_function_runtime() 