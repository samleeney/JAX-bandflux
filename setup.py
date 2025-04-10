from setuptools import setup, find_packages

setup(
    name="jax-bandflux",
    version="0.2.3",
    author="Samuel Alan Kossoff Leeney",
    author_email="sakl2@cam.ac.uk",
    description="A JAX-based package for calculating supernovae Bandfluxes",
    long_description=open("README.rst").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/samleeney/JAX-bandflux",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'jax_supernovae': [
            'data/models/salt3-nir/**/*',
            'sncosmo-modelfiles/**/*',
            'data/**/*'
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.10",
    install_requires=[
        'numpy>=1.24.0',
        'jax>=0.4.20',
        'jaxlib>=0.4.20',
        'sncosmo>=2.9.0',
        'pytest>=7.0.0',
        'pyyaml>=6.0.0',
        'matplotlib',
        'tqdm',
        'anesthetic',
        'requests'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'isort>=5.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinxcontrib-mermaid>=0.7.0',
            'sphinx-autodoc-typehints>=1.12.0',
        ],
    }
)
