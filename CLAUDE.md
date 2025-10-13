# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-bandflux is a JAX-based implementation of the SALT3-NIR supernova model for bandflux calculations. The codebase maintains exact consistency with SNCosmo while providing GPU acceleration and automatic differentiation capabilities through JAX.

## Essential Commands

### Development Setup
```bash
# Install in development mode with all dependencies
pip install -e .
pip install pytest numpy jax sncosmo pyyaml matplotlib

# IMPORTANT: For nested sampling, install the Handley Lab fork of BlackJAX
# This is NOT base BlackJAX - it includes nested sampling algorithms
# See: https://handley-lab.co.uk/nested-sampling-book/intro.html
pip install git+https://github.com/handley-lab/blackjax@proposal

pip install git+https://github.com/google-deepmind/distrax

# For development tools
pip install -e ".[dev]"  # Includes pytest, black, isort, sphinx
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_salt3nir_consistency.py -v  # CRITICAL: Run after any bandflux modifications
pytest tests/test_bandflux_performance.py -v  # Performance benchmarks
pytest tests/test_transmission_shifts.py -v   # Bandpass transmission tests
pytest tests/test_documentation.py -v         # Documentation code validation

# Run a single test function
pytest tests/test_salt3nir_consistency.py::test_function_name -v
```

### Code Formatting
```bash
# Format code with black
black jax_supernovae/

# Sort imports with isort
isort jax_supernovae/
```

### Documentation
```bash
# Build documentation locally
cd docs/
sphinx-build -b html . _build/html
```

## Architecture Overview

### Core Mathematical Engine (`jax_supernovae/salt3.py`)
The heart of the codebase implementing SALT3-NIR model calculations:
- **Primary Functions**: `salt3_m0`, `salt3_m1`, `salt3_colorlaw` - Core model components
- **Optimized Functions**: `optimized_salt3_bandflux`, `optimized_salt3_multiband_flux` - JIT-compiled high-performance versions
- **Bridge Pattern**: `precompute_bandflux_bridge` - Pre-computes integration grids for efficient multi-band calculations
- **Critical**: ALL mathematical operations use JAX (jnp), never numpy directly

### Bandpass System (`jax_supernovae/bandpasses.py`)
Sophisticated filter handling with JAX optimization:
- **Bandpass Class**: JAX-optimized interpolation with pre-computed integration grids
- **SVO Integration**: `create_bandpass_from_svo` for accessing Spanish Virtual Observatory filters
- **Transmission Shifts**: Support for wavelength-dependent transmission modifications
- **Registration System**: Global registry for efficient bandpass lookup

### Data Pipeline (`jax_supernovae/data.py`)
Handles Hubble Space Telescope DR1 format:
- **Main Function**: `load_and_process_data` - Unified data loading interface
- **Bridge Creation**: Automatic generation of optimized data structures
- **Multi-band Support**: Efficient handling of heterogeneous band observations
- **Redshift Hierarchy**: 
  1. Primary source: `data/redshifts.dat` - Contains high-quality spectroscopic redshifts
  2. Fallback source: `data/targets.dat` - Contains all targets with lower-quality redshifts
  3. Loading priority: Always use `redshifts.dat` if SN is present, otherwise fallback to `targets.dat`

### Testing Philosophy
The test suite serves as both validation and documentation:
- **Consistency Tests**: Ensure exact match with SNCosmo (within 0.001% tolerance)
- **Performance Tests**: Validate JAX speedups (typically >10x)
- **Integration Tests**: End-to-end nested sampling workflows
- **Documentation Tests**: All code examples must execute successfully

## Critical Implementation Rules

### JAX Compliance
1. **NEVER** use numpy in computational functions - only jax.numpy (jnp)
2. **ALWAYS** ensure functions are JIT-compilable (no Python control flow on array values)
3. **MAINTAIN** float64 precision: `jax.config.update("jax_enable_x64", True)`
4. **USE** `jnp.where` for conditional operations, not boolean indexing

### Model Consistency
1. **VERIFY** all bandflux modifications against SNCosmo using test_salt3nir_consistency.py
2. **MATCH** SNCosmo's integration grid exactly (5.0 Å spacing)
3. **PRESERVE** edge case handling in interpolation functions
4. **MAINTAIN** exact color law polynomial coefficients

### Performance Patterns
1. **PRECOMPUTE** bridge structures for repeated calculations
2. **VECTORIZE** operations across multiple bands/observations
3. **MINIMIZE** redundant interpolations
4. **LEVERAGE** JAX's automatic differentiation for optimization

## Data Structure Specifications

### Parameter Dictionary
```python
params = {
    'z': float,      # Redshift (0 to ~2)
    't0': float,     # Time of peak brightness (MJD)
    'x0': float,     # Amplitude (>0)
    'x1': float,     # Light-curve stretch (-3 to 3)
    'c': float       # Color parameter (-0.3 to 0.3)
}
```

### Bridge Structure (Performance Optimization)
```python
bridge = {
    'wave': jnp.array,   # Pre-computed wavelength grid
    'dwave': float,      # Grid spacing (5.0 Å)
    'trans': jnp.array   # Transmission values on grid
}
```

### Observation Format
- Times: Modified Julian Date (MJD)
- Fluxes: Measured flux values
- Flux Errors: Measurement uncertainties
- Zero Points: Photometric zero points (default 27.5)
- Band Indices: Integer indices into bridge array

## Common Workflows

### Loading and Fitting Data
```python
from jax_supernovae.data import load_and_process_data
from jax_supernovae.salt3 import optimized_salt3_multiband_flux
import jax

# Load data with fixed redshift
times, fluxes, fluxerrs, zps, band_indices, bridges, fixed_z = load_and_process_data('19agl', fix_z=True)

# Define likelihood function
@jax.jit
def loglikelihood(params):
    model_fluxes = optimized_salt3_multiband_flux(times, bridges, params, zps=zps, zpsys='ab')
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs)**2)
    return -0.5 * chi2
```

### Custom Bandpass Integration
```python
from jax_supernovae.bandpasses import create_bandpass_from_svo, register_bandpass

# Download and register UKIRT J-band filter
bandpass = create_bandpass_from_svo('UKIRT/WFCAM.J', output_dir='filter_data')
register_bandpass('ukirt_j', bandpass)
```

## AI Assistant Context (.airules)

The `.airules` file contains detailed implementation specifications and should be consulted for:
- Exact numerical algorithms and edge cases
- Testing requirements and validation procedures
- Error handling patterns
- Shape broadcasting rules

This file ensures AI assistants maintain consistency with the established codebase patterns and SNCosmo compatibility requirements.