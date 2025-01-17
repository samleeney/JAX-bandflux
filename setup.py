from setuptools import setup, find_packages

setup(
    name="jax_supernovae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "sncosmo"
    ],
    python_requires=">=3.8",
    author="Samuel Hinton",
    description="JAX implementation of supernova light curve models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/samuelreay/jax-supernovae",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    test_suite="tests",
    tests_require=["pytest>=6.0"],
    python_modules=["jax_supernovae"]
) 