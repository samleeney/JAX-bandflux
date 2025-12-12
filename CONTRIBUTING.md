# Contributing to JAX-bandflux

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
