# JAX-bandflux Documentation

This directory contains the documentation for JAX-bandflux, a Python package for differentiable supernova light curve modeling using JAX.

## Building the Documentation

To build the documentation, you need to have Sphinx and the required dependencies installed. You can install them using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Then, you can build the documentation using:

```bash
cd docs
make html
```

The built documentation will be available in the `_build/html` directory. You can open `_build/html/index.html` in your web browser to view it.

## Documentation Structure

The documentation is organized as follows:

- `index.rst`: Main entry point for the documentation
- `installation.rst`: Installation instructions
- `quickstart.rst`: Quickstart guide
- `core_concepts.rst`: Core concepts of JAX-bandflux
- `examples.rst`: Example scripts and notebooks
- `api/`: API reference documentation
- `tutorials/`: Step-by-step tutorials
- `guides/`: In-depth guides on specific topics

## Contributing to the Documentation

If you want to contribute to the documentation, please follow these guidelines:

1. Use reStructuredText (RST) format for documentation files
2. Follow the existing style and organization
3. Include examples and code snippets where appropriate
4. Test your changes by building the documentation locally

## ReadTheDocs Configuration

The documentation is built and hosted on ReadTheDocs. The configuration is in the `.readthedocs.yml` file in the root directory of the repository.