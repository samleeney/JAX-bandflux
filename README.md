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

This is a JAX implementation of the sncosmo bandflux function. The implementation is close to identical, but with a small change due to lack of availablity in JAX of an interpolation function. This leads to an ~0.001% different in bandflux results. Tests have been designed to confirm that after any changes, the match for key functions in the sncosmo version matches our implementation in JAX. If any changes are made to this section of the code these tests (which are automatically run as a github workflow) should be run to check that the matching is preserved. 