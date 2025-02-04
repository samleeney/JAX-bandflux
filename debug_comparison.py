import sncosmo
import numpy as np
from jax_supernovae.salt3_bandflux import salt3_m0, salt3_m1

# Create sncosmo model
model = sncosmo.Model(source='salt3-nir')

# Test points
phase = 5.0
wavelengths = np.linspace(3000.0, 9000.0, 10)

# Get values from sncosmo
m0_sncosmo = model._source._model['M0'](np.array([phase]), wavelengths)[0]

# Get values from our implementation
m0_ours = salt3_m0(phase, wavelengths)

print("Comparison at phase =", phase)
print("\nWavelengths:")
print(wavelengths)
print("\nSNCosmo M0 values:")
print(m0_sncosmo)
print("\nOur M0 values:")
print(m0_ours)
print("\nRatio (ours/sncosmo):")
print(m0_ours/m0_sncosmo)
print("\nAbsolute difference:")
print(np.abs(m0_ours - m0_sncosmo))
print("\nRelative difference (%):")
print(100 * np.abs(m0_ours - m0_sncosmo) / m0_sncosmo) 