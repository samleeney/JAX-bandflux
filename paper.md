---
title: 'JAX-based Differentiable Supernova Light Curve Modelling for Cosmological Analysis'
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
    affiliation: "1"
affiliations:
 - name: Cavendish Astrophysics, University of Cambridge, UK
   index: 1
date: "07 February 2025"
bibliography: paper.bib
---
# 1-2 line summary for editor
JAX-bandflux is a Python package for computing SALT model surfaces and bandflux for supernova light curve modeling using JAX in a differentiable framework. The pip installable package provides routines for data processing, bandpass registration, and model fitting for academic research.

# Summary

This paper describes a JAX [@jax2018github] implementation of critical supernova modelling functionality that is central to supernova light curve analysis. It replicates key components of the widely used non-differentiable SNCosmo [@barbary2016sncosmo] in a differentiable framework, enabling efficient parallelisation and gradient-based optimisation via GPUs. The codebase provides a differentiable approach to computing the multiband flux from a supernova's light curve, thereby facilitating the robust extraction of SALT parameters essential for cosmological analysis.

# Statement of need

Accurate estimation of supernova flux is a fundamental requirement in cosmological studies, as these measurements underpin the calibration of standard candles and hence distance determinations. Existing packages such as SNCosmo [@barbary2016sncosmo] have been extensively used in analysing supernova data; however, their traditional implementations are not differentiable. A differentiable approach is critical for efficiently propagating gradients during parameter optimisation and for performing large-scale parallel computations on modern hardware such as GPUs. This JAX implementation addresses these challenges by providing optimised, parallelisable routines that enable robust extraction of SALT parameters. Such improvements facilitate more rapid and reproducible cosmological analyses, as exemplified by recent works on SALT3 [@kenworthy2021salt3; @pierel2022salt3].

# Implementation

The codebase is organised into several modules and example scripts that showcase different aspects of the supernova modelling workflow. Two principal example scripts, `fmin_bfgs.py` and `ns.py`, demonstrate optimisation via L-BFGS-B and nested sampling respectively. Both scripts build on foundational routines provided in the JAX modules, thereby closely mirroring the SNCosmo architecture but enabling differentiability and efficient GPU acceleration. The central computation is contained in the file `salt3.py`, where the SALT3 model is implemented.

The SALT model is of the form:
$$
F(p, \lambda) = x_0 \left[ M_0(p, \lambda) + x_1 M_1(p, \lambda) + \ldots \right] \times \exp \left[ c \times CL(\lambda) \right]
$$
Where: free parameters are: $x_0$, $x_1$, $t_0$, and $c$. Model surface parameters are: $M_0(p, \lambda)$ and $M_1(p, \lambda)$ are functions that describe the underlying flux surfaces, and $p$ is a function of redshift and $t-2$.

The computation of the bandflux is achieved by integrating the model flux across the applied bandpass filters. Combining multiple bands, the bandflux is defined as:
$$
\text{bandflux} = \int_{\lambda_\text{min}}^{\lambda_\text{max}} F(\lambda) \cdot T(\lambda) \cdot \frac{\lambda}{hc} \, d\lambda
$$
Here, $T(\lambda)$ is the transmission function specific to the bandpass filter used; $h$ and $c$ are the Planck constant and the speed of light respectively.

Within `salt3.py`, the routine first computes the rest‐frame model flux by combining the base spectral surface $M_0(p, \lambda)$ with the stretch‐modulated variation $M_1(p, \lambda)$, each scaled by their respective SALT parameters. These operations are implemented using JAX's vectorised array manipulations, which are wrapped in JIT-compiled functions for efficient, parallel execution on GPUs. The resulting flux, expressed as 
$$
F(p, \lambda) = x_0 \left[ M_0(p, \lambda) + x_1 M_1(p, \lambda) + \ldots \right] \times \exp \left[ c \times CL(\lambda) \right],
$$ 
is computed in a fully differentiable manner. Next, the computed flux is multiplied by the instrument's transmission function $T(\lambda)$ and by the wavelength factor $\lambda/(hc)$—with $h$ and $c$ defined as the Planck constant and the speed of light respectively—using JAX's high-performance primitives. A trapezoidal integration is then applied along the wavelength dimension using a vectorised version of the integration (for example, via `jax.numpy.trapz`), which is also JIT-compiled and can be parallelised across multiple data instances with `vmap`. This sequence of operations produces a bandflux that is consistent with observational calibrations while taking full advantage of JAX's capabilities for optimised, parallel, and differentiable numerical computations.

The overall design allows gradient propagation through the entire pipeline thus enabling techniques that benefit from the differentiable, highly parrelisable programming paradigm provided by JAX. In doing so, the code replicates core functionalities of SNCosmo—traditionally implemented in a non-differentiable manner—while delivering increased efficiency and scalability for modern cosmological analyses.