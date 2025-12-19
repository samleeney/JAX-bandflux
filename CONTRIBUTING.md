# Contributing to JAX-bandflux

Thanks for your interest in contributing! This project is focused on differentiable SALT3 modelling with JAX. Please follow the guidelines below to help us review changes quickly.

## How to get help
- Open a GitHub issue for bugs, feature requests, or documentation clarifications.
- For usage questions, include a minimal code snippet and your JAX/JAXlib versions; note whether you are on CPU or GPU.

## Filing issues
- Include steps to reproduce, expected vs actual behaviour, and environment details (Python, JAX/JAXlib, CUDA/cuDNN if applicable).
- If performance is the concern, share a minimal benchmark and whether you are batching parameters or using `bandflux_batch`.

## Pull requests
- Target the latest development branch (e.g., `joss-reviews` or the active review branch).
- Keep PRs focused and small; describe the user-visible change and tests run.
- Run the test suite (`pytest`) before submitting; for GPU perf changes, include a note on the hardware used.

## Code style
- Prefer functional interfaces that are JIT-friendly; avoid hidden global state.
- Add concise comments only where the intent is non-obvious.
- Default to ASCII; avoid introducing non-ASCII unless necessary.

## Tests
- Add or update tests for new behaviour; ensure existing tests pass.
- Performance benchmarks should use the optimized paths (`bandflux` with bridges and `bandflux_batch` for batched params) to avoid legacy slow paths.

Thanks for helping improve JAX-bandflux!
