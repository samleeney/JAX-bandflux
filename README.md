# JAX Bandflux for Supernovae

A JAX-based implementation of supernova light curve modeling and analysis tools, focusing on anomaly detection in supernova data. This codebase provides efficient, differentiable implementations of core SNCosmo functionality using JAX, enabling advanced statistical analysis and machine learning applications.

## Installation

```bash
# Clone the repository
git clone https://github.com/samleeney/jax-supernovae.git
cd sn_anomaly

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Testing

This is a JAX implementation of the sncosmo bandflux function. The implementation is close to identical, but with a small change due to lack of availablity in JAX of a specific interpolation function. This leads to an ~0.001% different in bandflux results. Tests have been designed to confirm that after any changes, the match for key functions in the sncosmo version matches our implementation in JAX. If any changes are made to this section of the code these tests (which are automatically run as a github workflow) should be run to check that the matching is preserved. 

## Configuration Settings

The application's parameters are configured through a `settings.yaml` file in the root directory. This file contains all the hyperparameters for the nested sampling algorithm and prior distributions.

### Nested Sampling Parameters

```yaml
nested_sampling:
  n_live: 100        # Number of live points
  n_params: 5        # Number of parameters being sampled
  n_delete: 1        # Number of points to delete in each iteration
  num_mcmc_steps_multiplier: 5  # Multiplied by n_params for total MCMC steps
  max_iterations: 10000  # Maximum number of iterations
```

### Prior Bounds

The prior bounds section defines the minimum and maximum values for each parameter:

```yaml
prior_bounds:
  z:  # Redshift
    min: 0.001
    max: 0.2
  t0:  # Time of peak brightness
    min: 58000.0
    max: 59000.0
  x0:  # Amplitude (in log10 scale)
    min: -6.0
    max: -3.0
  x1:  # Light-curve stretch
    min: -3.0
    max: 3.0
  c:   # Color
    min: -0.3
    max: 0.3
```

### Prior Distributions

Some parameters use normal distributions as priors. These are configured as follows:

```yaml
prior_distributions:
  t0:
    type: normal
    loc: 58520.0    # Mean
    scale: 2.0      # Standard deviation
  x1:
    type: normal
    loc: 1.5
    scale: 1.0
  c:
    type: normal
    loc: 0.2
    scale: 0.2
```

Parameters not listed in the `prior_distributions` section use uniform priors based on their bounds defined in `prior_bounds`.

### Modifying Settings

To modify the behavior of the sampling:

1. Open `settings.yaml` in your preferred text editor
2. Adjust the relevant parameters
3. Save the file
4. Run the main script which will automatically use the new settings

Note: Ensure that the prior bounds and distributions are physically meaningful for your specific supernova analysis case. 