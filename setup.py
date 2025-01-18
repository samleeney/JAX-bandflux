from setuptools import setup, find_packages

setup(
    name="jax_supernovae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "numpy",
        "sncosmo",
        "requests>=2.25.0",
    ],
    python_requires=">=3.7",
)
