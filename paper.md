---
title: 'JAX-bandflux: differentiable supernovae SALT modelling for cosmological analysis on GPUs'
tags:
  - Python
  - astronomy
  - supernovae
  - cosmology
  - JAX
authors:
  - name: Samuel Alan Kossoff Leeney
    orcid: 0000-0003-4366-1119
    equal-contrib: true
    affiliation: "1, 2"
affiliations:
 - name: Astrophysics Group, Cavendish Laboratory, J. J. Thomson Avenue, Cambridge CB3 0HE, UK
   index: 1
 - name: Kavli Institute for Cosmology, Madingley Road, Cambridge CB3 0HA, UK
   index: 2
date: 9 April 2025
bibliography: paper.bib
---
# Summary

[JAX-bandflux](https://github.com/samleeney/JAX-bandflux) is a JAX [@jax2018github] implementation of critical supernova modelling functionality for cosmological analysis. The codebase implements key components of the established library SNCosmo [@barbary2016sncosmo] in a differentiable framework, offering efficient parallelisation and gradient-based optimisation capabilities through GPU acceleration. The package facilitates differentiable computation of supernova light curve measurements, supporting the inference of SALT [@kenworthy2021salt3; @pierel2022salt3] parameters necessary for cosmological analysis.

# Statement of need

Accurate estimation of supernova flux is essential in cosmological studies. These measurements are fundamental to the calibration of standard candles and subsequent distance determinations, which are used to answer cosmological questions. For example, the rate of expansion of the universe. Current packages such as SNCosmo [@barbary2016sncosmo] are widely used for analysing supernova data. However, traditional implementations are not designed to run on GPUs and they lack differentiability. A differentiable approach enables efficient gradient propagation during parameter optimisation and supports large-scale parallel computations on modern hardware such as GPUs. This JAX implementation addresses these requirements by providing differentiable, parallelisable routines for SALT parameter extraction.

# Implementation

The package is structured into several modules and example scripts that demonstrate various aspects of the supernova modelling workflow. Two primary example scripts, `fmin_bfgs.py` and `ns.py`, illustrate optimisation via L-BFGS-B and nested sampling respectively. These scripts utilise core routines from the JAX modules, following a structure similar to SNCosmo while enabling differentiability and GPU acceleration. The central computation is contained in the file `salt3.py`, which implements the SALT3 model.

The SALT model is of the form:
$$
F(p, \lambda) = x_0 \left[ M_0(p, \lambda) + x_1 M_1(p, \lambda) + \ldots \right] \times \exp \left[ c \times CL(\lambda) \right]
$$
where free parameters are: $x_0$, $x_1$, $t_0$, and $c$. Model surface parameters are: $M_0(p, \lambda)$ and $M_1(p, \lambda)$ are functions that describe the underlying flux surfaces, and $p$ is a function of redshift and $t-2$.

The computation of the bandflux is achieved by integrating the model flux across the applied bandpass filters. Combining multiple bands, the bandflux is defined as:
$$
\text{bandflux} = \int_{\lambda_\text{min}}^{\lambda_\text{max}} F(\lambda) \cdot T(\lambda) \cdot \frac{\lambda}{hc} \, d\lambda
$$
Here, $T(\lambda)$ is the transmission function specific to the bandpass filter used; $h$ and $c$ are the Planck constant and the speed of light respectively.

Within `salt3.py`, the implementation computes the rest-frame model flux by combining the base spectral surface $M_0(p, \lambda)$ with the stretch-modulated variation $M_1(p, \lambda)$, each scaled by their respective SALT parameters. These operations utilise JAX's vectorised array manipulations, which are JIT-compiled for efficient, parallel execution on GPUs. The resulting flux is computed in a fully differentiable manner. The computed flux is then multiplied by the instrument's transmission function $T(\lambda)$ and by the wavelength factor $\lambda/(hc)$, followed by trapezoidal integration along the wavelength dimension using JAX's numerical integration capabilities. These operations are also JIT-compiled and can be parallelised across multiple data instances via `vmap`.

The package includes comprehensive bandpass filter handling through the `bandpasses.py` module, which provides a `Bandpass` class to represent filter transmission functions. A set of commonly used astronomical filters is pre-integrated into the system, whilst additional custom bandpasses can be registered as needed through functions such as `register_bandpass` and `load_bandpass_from_file`. The system also facilitates the creation of bandpass objects from the Spanish Virtual Observatory (SVO) filter service. For data handling, the `data.py` module offers utilities for loading and processing supernova observations from various formats, including functions to handle redshift data and prepare it for model fitting. The package currently supports both SALT3 and SALT3-NIR models through dedicated interpolation routines found in `salt3.py`.

This architecture allows gradient propagation through the entire analysis pipeline, enabling techniques that benefit from JAX's differentiable, parallelisable programming paradigm. The implementation maintains functional parity with SNCosmo whilst providing an enhanced computational efficiency and scalability for contemporary cosmological analyses.