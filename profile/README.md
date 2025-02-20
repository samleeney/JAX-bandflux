# SALT3 Profiling

This directory contains tools for profiling the SALT3 JAX implementation.

## Requirements

To run the profiling scripts, you need:

- JAX with CUDA support (for GPU profiling)
- TensorBoard (for viewing profiling results)
- numpy

## Running the Profiler

To run the profiling script:

```bash
python profile_salt3.py
```

The script will:
1. Create test data for profiling
2. Profile individual SALT3 components (M0, M1, color law)
3. Profile flux calculations (single band and multiband)
4. Save profiling data to a temporary directory

## Viewing Results

The script will output the location of the profiling data. You can view the results using TensorBoard:

```bash
tensorboard --logdir=/path/to/profile/data
```

This will start a TensorBoard server that you can access in your web browser.

## Understanding the Results

The profiler measures:
- Computation time for individual components (M0, M1, color law)
- Computation time for flux calculations
- Memory usage
- XLA/JAX operations and their timings

The results can help identify:
- Performance bottlenecks
- Memory usage patterns
- Opportunities for optimisation
- JIT compilation overhead 