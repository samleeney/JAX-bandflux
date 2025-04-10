Core Concepts
============

This page explains the core concepts of JAX-bandflux and how they relate to supernova light curve modeling. Understanding these concepts is essential for effectively using JAX-bandflux for your research.

.. contents:: Table of Contents
   :local:
   :depth: 2
SALT3 Model
----------

The SALT3 (Spectral Adaptive Lightcurve Template 3) model is a parametric model for Type Ia supernova light curves. It extends the SALT2 model with improved near-infrared coverage and updated training data.

### Mathematical Formulation

The SALT model is of the form:

.. math::

   F(p, \lambda) = x_0 \left[ M_0(p, \lambda) + x_1 M_1(p, \lambda) + \ldots \right] \times \exp \left[ c \times CL(\lambda) \right]

Where:

* :math:`x_0` is the overall flux normalization
* :math:`x_1` is the stretch parameter
* :math:`t_0` is the time of peak brightness
* :math:`c` is the color parameter
* :math:`M_0(p, \lambda)` and :math:`M_1(p, \lambda)` are functions that describe the underlying flux surfaces
* :math:`p` is a function of redshift and :math:`t-t_0`
* :math:`CL(\lambda)` is the color law

### Model Components

The SALT3 model consists of several key components:

1. **M0 Component**: The mean spectral energy distribution (SED) of a Type Ia supernova as a function of phase and wavelength. This represents the "average" SN Ia.

2. **M1 Component**: The first-order variation in the SED, which captures the primary spectral diversity of SNe Ia. This is multiplied by the stretch parameter x1.

3. **Color Law**: A function that describes how the color parameter c affects the SED at different wavelengths. It's implemented as a polynomial in wavelength.

4. **Redshift Correction**: The model accounts for redshift by scaling the wavelengths and applying a flux normalization factor.

### Parameter Interpretation

* **x0**: Controls the overall brightness of the supernova. It's related to the distance modulus and is typically in the range of 10^-5 to 10^-4.

* **x1**: The "stretch" parameter, which correlates with the width of the light curve. Positive values indicate broader light curves (slower decline), while negative values indicate narrower light curves (faster decline). Typically ranges from -3 to 3.

* **c**: The color parameter, which correlates with the B-V color of the supernova. Positive values indicate redder colors, while negative values indicate bluer colors. Typically ranges from -0.3 to 0.3.

* **t0**: The time of peak brightness in the B-band, measured in Modified Julian Date (MJD).

* **z**: The redshift of the supernova, which affects both the observed wavelengths and the time dilation.

### Visual Representation

The following figure illustrates how the SALT3 model parameters affect the light curve:

.. figure:: https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/docs/_static/salt3_parameters.png
   :alt: SALT3 Parameter Effects
   :width: 80%
   :align: center

   Effect of SALT3 parameters on light curves. Left: Effect of x1 (stretch). Right: Effect of c (color).

JAX-bandflux implements the SALT3 model in a differentiable way using JAX, which enables efficient gradient-based optimization and GPU acceleration.
JAX-bandflux implements the SALT3 model in a differentiable way using JAX, which enables efficient gradient-based optimization and GPU acceleration.
Bandpass Management
-----------------

Bandpasses represent the transmission functions of astronomical filters. They define how much light at each wavelength passes through the filter.

### Bandpass Integration

The computation of the bandflux is achieved by integrating the model flux across the applied bandpass filters:

.. math::

   \text{bandflux} = \int_{\lambda_\text{min}}^{\lambda_\text{max}} F(\lambda) \cdot T(\lambda) \cdot \frac{\lambda}{hc} \, d\lambda

Where:

* :math:`F(\lambda)` is the model flux as a function of wavelength
* :math:`T(\lambda)` is the transmission function of the bandpass filter
* :math:`\lambda` is the wavelength
* :math:`h` is the Planck constant
* :math:`c` is the speed of light

The factor :math:`\frac{\lambda}{hc}` converts from energy flux to photon flux, which is what most astronomical detectors measure.

### Bandpass Implementation

In JAX-bandflux, bandpasses are implemented as the `Bandpass` class, which contains:

* A wavelength array (`wave`)
* A transmission array (`trans`)
* Methods for interpolation and integration

The integration is performed numerically using the trapezoidal rule on a pre-computed wavelength grid, which is optimized for accuracy and performance.

### Precomputed Bridge Data

For efficiency, JAX-bandflux precomputes "bridge data" for each bandpass, which includes:

* The integration wavelength grid
* The spacing between grid points
* The transmission values on the grid

This precomputed data is passed to the flux calculation functions, avoiding redundant computations and improving performance.

### Bandpass Management Features

JAX-bandflux provides a flexible system for managing bandpasses:

* **Built-in Support**: Common astronomical filters (ZTF, ATLAS, SDSS, 2MASS, WFCAM)
* **Custom Bandpasses**: Ability to register custom bandpasses from files or arrays
* **SVO Integration**: Access to the Spanish Virtual Observatory (SVO) Filter Profile Service
* **Bandpass Registry**: A global registry for managing bandpasses by name
* **Bandpass Manipulation**: Functions for combining, modifying, and visualizing bandpasses

### Visual Representation

The following figure illustrates the bandpass integration process:

.. figure:: https://raw.githubusercontent.com/samleeney/JAX-bandflux/master/docs/_static/bandpass_integration.png
   :alt: Bandpass Integration
   :width: 80%
   :align: center

   Bandpass integration process. The model flux (blue) is multiplied by the bandpass transmission (orange) to produce the integrated flux.
* Integration with the Spanish Virtual Observatory (SVO) Filter Profile Service

Data Handling
-----------

JAX-bandflux provides utilities for loading and processing supernova light curve data, making it easy to work with real observations.

### Data Formats

JAX-bandflux primarily supports the Hierarchical Supernova Format (HSF), which is a standardized format for supernova light curve data. The data is typically organized as:

* A directory for each supernova (e.g., `data/19dwz/`)
* A photometry file containing observations (e.g., `all.phot`)
* Optional metadata files (e.g., `redshifts.dat`)

The photometry file contains columns for:

* Time (MJD)
* Band/filter name
* Flux
* Flux error
* Zero point
* Zero point system

### Data Loading Process

The data loading process in JAX-bandflux involves several steps:

1. **File Discovery**: Finding the appropriate data files for a given supernova
2. **Parsing**: Reading the data files and extracting the relevant information
3. **Validation**: Checking that the required columns are present
4. **Preprocessing**: Converting to JAX arrays, handling missing values, etc.
5. **Bandpass Registration**: Registering the bandpasses needed for the data
6. **Bridge Computation**: Precomputing bridge data for efficient flux calculations

### Key Data Functions

* `load_hsf_data`: Loads raw data from HSF format files
* `load_redshift`: Loads redshift information from a redshifts.dat file
* `load_and_process_data`: Combines the above steps into a single function that returns all the data needed for modeling

### Example Data Structure

```python
# Example of data returned by load_and_process_data
times = jnp.array([58650.1, 58651.2, ...])  # Observation times
fluxes = jnp.array([1.2e-5, 1.3e-5, ...])   # Observed fluxes
fluxerrs = jnp.array([1.0e-6, 1.1e-6, ...]) # Flux errors
zps = jnp.array([27.5, 27.5, ...])          # Zero points
band_indices = jnp.array([0, 1, 0, ...])    # Indices into bridges
bridges = (bridge0, bridge1, ...)           # Precomputed bridge data
fixed_z = (0.1234, 0.0012)                  # Redshift and error
```

Differentiable Programming with JAX
---------------------------------

JAX-bandflux leverages JAX for differentiable programming, which provides several key advantages for supernova light curve modeling.

### Automatic Differentiation

JAX provides automatic differentiation, which allows for efficient computation of gradients. This is particularly useful for:

* **Gradient-based optimization**: Finding the best-fit parameters using methods like L-BFGS-B
* **Sensitivity analysis**: Understanding how changes in parameters affect the model
* **Fisher matrix calculations**: Estimating parameter uncertainties

### Just-in-Time (JIT) Compilation

JAX's JIT compilation converts Python functions into optimized machine code, providing significant performance improvements:

* **Reduced overhead**: Eliminates Python interpretation overhead
* **Optimized execution**: Applies compiler optimizations
* **Efficient memory usage**: Minimizes memory allocations

Example of JIT compilation:

```python
@jax.jit
def salt3_bandflux(phase, bandpass, params):
    # Function implementation
    ...
```

### Vectorization

JAX provides vectorization through its `vmap` function, which allows for efficient parallel computation:

* **Batch processing**: Process multiple observations at once
* **Parameter exploration**: Evaluate the model for multiple parameter sets
* **Efficient use of hardware**: Utilize SIMD instructions on CPU or GPU

Example of vectorization:

```python
# Vectorize over phases
phase_vmap = jax.vmap(lambda p: salt3_bandflux(p, bandpass, params))(phases)
```

### GPU Acceleration

JAX can automatically run computations on GPUs, providing significant speedups for:

* **Large-scale analyses**: Processing many supernovae
* **Complex models**: Models with many parameters or components
* **Monte Carlo simulations**: Generating and analyzing many simulated datasets

### Performance Comparison

The following table shows typical performance improvements with JAX:

| Operation | NumPy | JAX (CPU) | JAX (GPU) |
|-----------|-------|-----------|-----------|
| Single flux calculation | 1x | 5-10x | 20-50x |
| Batch flux calculation | 1x | 10-20x | 50-100x |
| Gradient computation | N/A | 1-2x overhead | 1-2x overhead |

These capabilities make JAX-bandflux well-suited for large-scale supernova cosmology analyses, where performance and scalability are critical.
Model Fitting
-----------

JAX-bandflux supports various approaches to fitting SALT parameters to supernova light curve data, allowing for both frequentist and Bayesian analyses.

### Gradient-Based Optimization

JAX-bandflux is particularly well-suited for gradient-based optimization methods due to its differentiable nature. The most commonly used method is L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds), which:

* Uses gradient information to efficiently navigate the parameter space
* Handles parameter bounds to ensure physically meaningful results
* Converges quickly for well-behaved problems

Example workflow:

1. Define an objective function (typically chi-squared)
2. Set initial parameter values and bounds
3. Use `scipy.optimize.minimize` with the L-BFGS-B method
4. Extract the best-fit parameters and uncertainties

### Nested Sampling

For Bayesian inference, JAX-bandflux supports nested sampling through integration with the `blackjax` library. Nested sampling:

* Samples the posterior distribution of parameters
* Computes the Bayesian evidence (marginal likelihood)
* Handles multimodal distributions and complex parameter spaces

Example workflow:

1. Define a likelihood function and prior distributions
2. Set up the nested sampling algorithm
3. Run the sampler to generate posterior samples
4. Analyze the results using corner plots and summary statistics

### Custom Optimization Algorithms

JAX-bandflux's modular design allows for integration with custom optimization algorithms:

* **Markov Chain Monte Carlo (MCMC)**: For sampling posterior distributions
* **Particle Swarm Optimization**: For global optimization without gradients
* **Genetic Algorithms**: For complex, non-convex optimization problems

### Comparison of Fitting Methods

| Method | Advantages | Disadvantages | Use Cases |
|--------|------------|---------------|-----------|
| L-BFGS-B | Fast, uses gradients, handles bounds | Can get stuck in local minima | Quick parameter estimation, well-behaved problems |
| Nested Sampling | Computes evidence, handles multimodality | Computationally intensive | Model comparison, complex parameter spaces |
| MCMC | Samples posterior, handles correlations | Slow convergence, tuning required | Detailed posterior analysis, non-Gaussian uncertainties |

The differentiable nature of JAX-bandflux makes it particularly well-suited for gradient-based optimization methods, but its flexibility allows for a wide range of fitting approaches.

### Model Comparison and Selection

JAX-bandflux facilitates model comparison and selection through:

* **Chi-squared statistics**: For frequentist model comparison
* **Bayesian evidence**: For Bayesian model comparison
* **Information criteria**: AIC, BIC for balancing goodness-of-fit and model complexity
The differentiable nature of JAX-bandflux makes it particularly well-suited for gradient-based optimization methods.