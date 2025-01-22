import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import sncosmo
import matplotlib.pyplot as plt

# Load data and set up model
model = sncosmo.Model(source='salt2')
data = sncosmo.load_example_data()

# Define an objective function that we will pass to the minimizer.
# The function arguments must comply with the expectations of the specfic
# minimizer you are using.
def objective(parameters):
    model.parameters[:] = parameters  # set model parameters

    # evaluate model fluxes at times/bandpasses of data
    model_flux = model.bandflux(data['band'], data['time'],
                                zp=data['zp'], zpsys=data['zpsys'])

    jax_flux = salt3nir_bandflux(data['time'], data['band'], parameters)

    # calculate and return chi^2
    return np.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)

# starting parameter values in same order as `model.param_names`:
start_parameters = [0.4, 55098., 1e-5, 0., 0.]  # z, t0, x0, x1, c

# parameter bounds in same order as `model.param_names`:
bounds = [(0.3, 0.7), (55080., 55120.), (None, None), (None, None),
          (None, None)]

parameters, val, info = fmin_l_bfgs_b(objective, start_parameters,
                                      bounds=bounds, approx_grad=True)

# Set model parameters to best fit
model.parameters = parameters

# Plot the light curve
plt.figure(figsize=(10, 6))

# Define a color map for the bands
colors = {'sdssg': 'g', 'sdssr': 'r', 'sdssi': 'purple', 'sdssz': 'k'}

# Plot data points and model curves for each band
times = np.linspace(data['time'].min(), data['time'].max(), 100)

for band in np.unique(data['band']):
    color = colors[band]
    
    # Plot data points
    mask = data['band'] == band
    plt.errorbar(data['time'][mask], data['flux'][mask],
                yerr=data['fluxerr'][mask],
                fmt='o', color=color, label=f'Data {band}')
    
    # Plot model curve with same color
    model_flux = model.bandflux(band, times,
                               zp=data['zp'][0], zpsys=data['zpsys'][0])
    plt.plot(times, model_flux, '-', color=color, label=f'Model {band}')

plt.xlabel('Time (MJD)')
plt.ylabel('Flux')
plt.title('SN Light Curve - Data and Best Fit Model')
plt.legend()
plt.grid(True)
plt.savefig('sncosmo-fitter.png')

# Print best-fit parameters
print("\nBest-fit parameters:")
for name, value in zip(model.param_names, parameters):
    print(f"{name}: {value:.6f}")