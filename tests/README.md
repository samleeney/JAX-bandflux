# JAX-bandflux Tests

This directory contains tests for the JAX-bandflux package.

## Test Types

### Unit Tests
- `test_all.py`: Entry point for running all tests
- `test_bandflux_performance.py`: Tests for bandflux performance
- `test_ns_completion.py`: Tests for nested sampling completion
- `test_salt3nir_consistency.py`: Tests for SALT3-NIR model consistency

### Documentation Tests
- `test_documentation.py`: Tests for code blocks in the ReadTheDocs documentation

The documentation tests verify that all code examples in the documentation are functional and produce the expected results. These tests cover:

1. **Quickstart Guide**: Basic data loading and model flux calculation
2. **Data Loading**: Synthetic and real data handling
3. **Bandpass Loading**: Custom bandpass creation and registration
4. **Model Fluxes**: SALT3 parameter definition and flux calculation
5. **Sampling**: Objective function definition and optimization

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run only documentation tests:
```bash
pytest tests/test_documentation.py
```

## CI Integration

These tests are automatically run in the GitHub Actions workflow defined in `.github/workflows/test.yml`.