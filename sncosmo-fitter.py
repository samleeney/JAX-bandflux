import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import sncosmo
import matplotlib.pyplot as plt
import astropy.table as at
from astropy.io import ascii
import requests
from io import StringIO

def load_filter_from_svo(filter_id):
    """Load a filter from the Spanish Virtual Observatory (SVO) Filter Profile Service."""
    base_url = "http://svo2.cab.inta-csic.es/theory/fps/fps.php"
    params = {
        "format": "ascii",
        "ID": filter_id
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to load filter {filter_id}")
    
    # Skip header lines and parse data
    lines = response.text.split('\n')
    data_lines = []
    for line in lines:
        if line.startswith('#'):
            continue
        if not line.strip():
            continue
        data_lines.append(line)
    
    # Parse wavelength and transmission
    wave = []
    trans = []
    for line in data_lines:
        try:
            w, t = map(float, line.split()[:2])
            wave.append(w)
            trans.append(t)
        except (ValueError, IndexError):
            continue
    
    if not wave:
        raise Exception(f"No valid data found for filter {filter_id}")
    
    # Convert wavelength from nm to Å
    wave = np.array(wave) * 10
    trans = np.array(trans)
    
    # Create and register bandpass
    band = sncosmo.Bandpass(wave, trans, name=filter_id)
    sncosmo.register(band, force=True)
    return band

# Load and register required bandpasses
def register_hsf_bandpasses():
    # Load WFCAM J-band filter for J_1D3 and J_2D
    j_band = load_filter_from_svo("UKIRT/WFCAM.J_WFCAM")
    sncosmo.register(j_band, 'J_1D3', force=True)
    sncosmo.register(j_band, 'J_2D', force=True)
    
    # Load ATLAS cyan filter
    c_band = load_filter_from_svo("ATLAS/ATLAS.c")
    sncosmo.register(c_band, 'c', force=True)
    
    # Load ATLAS orange filter
    o_band = load_filter_from_svo("ATLAS/ATLAS.o")
    sncosmo.register(o_band, 'o', force=True)

# Load data from HSF_DR1
def load_hsf_data(sn_name):
    data = at.Table.read(f'hsf_DR1/Ia/{sn_name}/all.phot', format='ascii')
    # Rename columns to match sncosmo requirements
    data.rename_column('bandpass', 'band')
    data.rename_column('mjd', 'time')
    return data

# Register the required bandpasses
register_hsf_bandpasses()

# Load data and set up model
model = sncosmo.Model(source='salt3-nir')
data = load_hsf_data('22nbn')  # Using 22nbn as an example

# Define an objective function that we will pass to the minimizer.
def objective(parameters):
    model.parameters[:] = parameters  # set model parameters

    # evaluate model fluxes at times/bandpasses of data
    model_flux = model.bandflux(data['band'], data['time'],
                               zp=data['zp'], zpsys=data['zpsys'])

    # calculate and return chi^2
    return np.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)

# Known redshift for SN 22nbn
z = 0.04514918889418029

# starting parameter values in same order as `model.param_names`:
# [z, t0, x0, x1, c]
t0_guess = np.mean(data['time'])  # rough guess for peak time
start_parameters = [z, t0_guess, 1e-4, 0., 0.]

# parameter bounds in same order as `model.param_names`:
bounds = [(z*0.999, z*1.001),  # very tight bound around known redshift
         (t0_guess-20, t0_guess+20),  # peak within ±20 days of guess
         (1e-6, 1e-2),  # x0 (amplitude)
         (-3., 3.),  # x1 (stretch)
         (-0.3, 0.3)]  # c (color)

parameters, val, info = fmin_l_bfgs_b(objective, start_parameters,
                                     bounds=bounds, approx_grad=True)

# Set model parameters to best fit
model.parameters = parameters

# Plot the light curve
plt.figure(figsize=(12, 8))

# Plot data points and model curves for each band
times = np.linspace(data['time'].min(), data['time'].max(), 100)

# Define colors for each band
colors = {
    'c': 'cyan',
    'o': 'orange',
    'ztfg': 'g',
    'ztfr': 'r',
    'J_1D3': 'purple',
    'J_2D': 'magenta'
}

for band in np.unique(data['band']):
    color = colors.get(band, 'k')
    # Plot data points
    mask = data['band'] == band
    plt.errorbar(data['time'][mask], data['flux'][mask],
                yerr=data['fluxerr'][mask],
                fmt='o', color=color, label=f'Data {band}')
    
    # Plot model curve
    model_flux = model.bandflux(band, times,
                               zp=data['zp'][0], zpsys=data['zpsys'][0])
    plt.plot(times, model_flux, '-', color=color, label=f'Model {band}')

plt.xlabel('Time (MJD)')
plt.ylabel('Flux')
plt.title('SN 22nbn Light Curve - Data and SALT3-NIR Model Fit')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('sncosmo-fitter.png', bbox_inches='tight')

# Print best-fit parameters
print("\nBest-fit parameters:")
for name, value in zip(model.param_names, parameters):
    print(f"{name}: {value:.6f}")

# Print chi-square value
print(f"\nFinal chi-square: {val:.2f}")
print(f"Number of data points: {len(data)}")
print(f"Reduced chi-square: {val/(len(data)-len(parameters)):.2f}")