import jax
import time

print(f"Running traced computation...")
with jax.profiler.trace(base_dir, create_perfetto_link=False):
    # Run multiple evaluations to see the performance
    for i in range(100):
        with jax.profiler.TraceAnnotation(f"PROFILE_iteration_{i}"):
            result = compute_single_loglikelihood(
                test_params, times, fluxes, fluxerrs, zps, band_indices, bridges
            )
            result.block_until_ready()

# Wait a moment for the trace to be written
time.sleep(1) 