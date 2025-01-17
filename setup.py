from setuptools import setup, find_packages

setup(
    name="jax_supernovae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "numpy",
        "sncosmo",
    ],
    python_requires=">=3.7",
)
