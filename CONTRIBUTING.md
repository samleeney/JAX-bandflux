# Contributing to JAX-bandflux

<<<<<<< HEAD
Thanks for your interest in contributing! This project is focused on differentiable SALT3 modelling with JAX. Please follow the guidelines below to help us review changes quickly and keep discussions constructive.

## How to get help
- Open a GitHub issue for bugs, feature requests, or documentation clarifications (fork off the latest `main`/`joss-reviews` when sending a PR).
- For usage questions, include a minimal code snippet and your JAX/JAXlib versions; note whether you are on CPU or GPU.

## Filing issues
- Include steps to reproduce, expected vs actual behaviour, and environment details (Python, JAX/JAXlib, CUDA/cuDNN if applicable).
- If performance is the concern, share a minimal benchmark and whether you are batching parameters or using `bandflux_batch`.

## Pull requests
- Target the latest development branch (e.g., `joss-reviews` or the active review branch).
- Keep PRs focused and small; describe the user-visible change and tests run.
- Run the test suite (`pytest`) before submitting; for GPU perf changes, include a note on the hardware used.

## Code style and docs
- Prefer functional interfaces that are JIT-friendly; avoid hidden global state.
- Follow existing style; format with `black`/`isort` if applicable and keep docstrings clear for autodoc.
- Add concise comments only where the intent is non-obvious.
- Default to ASCII; avoid introducing non-ASCII unless necessary.
- Update README/docs for user-facing changes; autodoc pages surface public APIs.

## Tests
- Add or update tests for new behaviour; ensure existing tests pass.
- Performance benchmarks should use the optimized paths (`bandflux` with bridges and `bandflux_batch` for batched params) to avoid legacy slow paths.

## Code of Conduct
- Be respectful and inclusive; this project follows the Contributor Covenant.

Thanks for helping improve JAX-bandflux!
=======
Thanks for your interest in contributing! This guide covers the basics for reporting issues, submitting pull requests, and getting help.

## Getting help / questions
- Open a GitHub issue with a clear description of the problem or question, including environment (Python/JAX version), install path (PyPI vs GitHub), and a minimal snippet to reproduce.
- Discussions are welcome in issues; please keep the tone constructive and follow the Code of Conduct.

## Reporting bugs
- Check existing issues first to avoid duplicates.
- Include steps to reproduce, expected vs. actual behavior, stack traces, and any data needed to reproduce (or a small synthetic example).
- Note whether you installed via PyPI, GitHub, or editable install.

## Feature requests
- Describe the use case and any alternatives you considered.
- If it’s API-related, propose a minimal signature/usage example.

## Pull requests
- Fork and create a branch off the latest `main` (or `joss-reviews` if addressing review items).
- Keep changes focused; smaller PRs are reviewed faster.
- Add or update tests/docs when behavior changes.
- Ensure `pytest` passes locally if possible; CI will run the test suite.
- Link the PR to the related issue and include a short summary of the change.

## Coding style
- Follow existing style; format Python code with `black` and sort imports with `isort` where applicable.
- Prefer explicit imports and clear docstrings for public functions.

## Documentation
- Update README and docs for user-facing changes.
- For API changes, ensure docstrings are accurate; autodoc pages will surface them.

## Code of Conduct
- This project follows the JOSS/Contributor Covenant Code of Conduct. Be respectful and inclusive in all interactions.
>>>>>>> origin/joss-review-packaging
