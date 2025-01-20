from setuptools import setup, find_packages

setup(
    name="jax_supernovae",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'jax',
        'sncosmo',
    ],
)
