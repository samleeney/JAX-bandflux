import numpy as np
from jax_supernovae.core import Bandpass, MagSystem
from jax_supernovae.models import Model
from jax_supernovae.salt2 import salt2_flux
import sncosmo
from sncosmo.bandpasses import Bandpass as SNCBandpass

# Create SNCosmo model
sn_model = sncosmo.Model(source='salt2')
sn_model.set(z=0.1, t0=0.0, x0=1e-5, x1=0.1, c=0.2)

# Create JAX model with the same parameters
params = {'z': 0.1, 't0': 0.0, 'x0': 1e-5, 'x1': 0.1, 'c': 0.2}
jax_model = Model(source=None)
jax_model.flux = lambda t, w: salt2_flux(t, w, params)
jax_model.parameters = params

# Test flux at a single point
t = 0.0
w = 5000.0
print(f"\nTesting flux at single point:")
print(f"t = {t}, w = {w}\n")

# Create a test bandpass with 101 points centered on w
wave = np.linspace(w - 500, w + 500, 101)
trans = np.zeros_like(wave)
trans[50] = 1.0  # Set transmission to 1.0 at the central point
test_band_snc = SNCBandpass(wave, trans)
test_band = Bandpass(wave, trans)

# Get flux from SNCosmo
sn_flux = sn_model.bandflux(test_band_snc, np.array([t]))[0]
print(f"\nSNCosmo flux:   {sn_flux:.3e}")

# Get flux from JAX
jax_flux = jax_model.bandflux(test_band, np.array([t]))[0]
print(f"JAX flux:      {jax_flux:.3e}")
print(f"Ratio:       {jax_flux/sn_flux:.3f}\n")

# Print intermediate values from SALT2 calculation
print("\nSALT2 calculation details:")
z = 0.1
a = 1.0 / (1.0 + z)
t_rest = t * a
w_rest = w * a

print("\nRest-frame values:")
print(f"t_rest = {t_rest}")
print(f"w_rest = {w_rest}")

# Get SNCosmo M0 and M1
sn_m0 = sn_model._source._model['M0'](np.array([t_rest]), np.array([w_rest]))[0][0]
sn_m1 = sn_model._source._model['M1'](np.array([t_rest]), np.array([w_rest]))[0][0]
sn_cl = sn_model._source._colorlaw(np.array([w_rest]))[0]

print("\nSNCosmo components:")
print(f"M0: {sn_m0:11.3e}")
print(f"M1: {sn_m1:11.3e}")
print(f"colorlaw: {sn_cl:11.3e}")

# Get JAX M0 and M1
from jax_supernovae.salt2 import salt2_m0, salt2_m1, salt2_colorlaw
jax_m0 = salt2_m0(t_rest, w_rest)
jax_m1 = salt2_m1(t_rest, w_rest)
jax_cl = salt2_colorlaw(w_rest)

print("\nJAX components:")
print(f"M0: {float(jax_m0):11.3e}")
print(f"M1: {float(jax_m1):11.3e}")
print(f"colorlaw: {float(jax_cl):11.3e}")

# Print parameters
x0 = params['x0']
x1 = params['x1']
c = params['c']

print("\nParameters:")
print(f"x0: {x0:11.3e}")
print(f"x1: {x1:11.3e}")
print(f"c:  {c:11.3e}")

# Compute final flux
sn_final = x0 * (sn_m0 + x1 * sn_m1) * 10**(-0.4 * sn_cl * c)
jax_final = x0 * (jax_m0 + x1 * jax_m1) * 10**(-0.4 * jax_cl * c)

print("\nFinal flux calculation:")
print(f"SNCosmo: {sn_final:11.3e}")
print(f"JAX:     {jax_final:11.3e}")
print(f"Ratio:   {jax_final/sn_final:.3f}") 