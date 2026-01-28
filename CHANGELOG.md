# Changelog

All notable changes to JAX-bandflux will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-28

### Added
- CITATION.cff for software citation
- Zenodo integration (.zenodo.json) for archival
- CHANGELOG.md for version history

### Changed
- Addressed JOSS reviewer feedback
- Improved documentation with doctest-validated examples
- Updated installation instructions to use uv

### Fixed
- Docstring formatting for proper Sphinx rendering
- ReadTheDocs build configuration

## [0.3.11] - 2026-01-27

### Changed
- Address JOSS reviewer feedback from @Samreay

## [0.3.10] - 2026-01-26

### Changed
- Update installation instructions to use uv
- Remove unused files and development notes

## [0.3.9] - 2026-01-25

### Fixed
- Docstring formatting for proper Sphinx rendering

## [0.3.8] - 2026-01-24

### Fixed
- RTD build - use Python 3.11 for tomllib support
- ReadTheDocs build - use [docs] extras instead of [dev]

## [0.3.7] - 2026-01-23

### Changed
- Rewrite documentation with doctest-validated examples and plots

## [0.3.6] - 2026-01-22

### Changed
- Read docs version dynamically from pyproject.toml

## [0.3.5] - 2026-01-21

### Fixed
- pyproject.toml syntax error - remove duplicate packages.find
- blackjax dependency for PyPI compatibility

## [0.3.0] - 2025-04-10

### Added
- Initial JOSS submission version
- SALT3 and SALT3-NIR model support
- GPU acceleration via JAX
- Comprehensive bandpass filter handling
- Example scripts for optimization and nested sampling

## [0.1.9] - 2025-03-04

### Added
- Custom bandpass implementation

## [0.1.8] - 2025-02-25

### Added
- JAX implementation of SALT3 bandflux (Alpha)

## [0.1.7] - 2025-02-06

### Added
- Initial release: JAX implementation of SALT3-NIR bandflux (Alpha)
