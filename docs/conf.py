# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import doctest
import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'JAX-bandflux'
copyright = '2025, Samuel Alan Kossoff Leeney'
author = 'Samuel Alan Kossoff Leeney'

# Read version dynamically from pyproject.toml
pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject = tomllib.load(f)
version = pyproject["project"]["version"]
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx_rtd_theme',
    'sphinxcontrib.mermaid',
    'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Add custom CSS
html_css_files = [
    'custom.css',
]

# -- Extension configuration -------------------------------------------------
# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# AutoDoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Doctest configuration ---------------------------------------------------
# Global setup code run before every doctest block
doctest_global_setup = '''
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax_supernovae import SALT3Source, TimeSeriesSource
from jax_supernovae.bandpasses import get_bandpass, Bandpass, register_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge
'''

# Doctest flags for flexible output matching
doctest_default_flags = (
    doctest.ELLIPSIS |
    doctest.NORMALIZE_WHITESPACE |
    doctest.IGNORE_EXCEPTION_DETAIL
)

# -- Plot directive configuration --------------------------------------------
# Include source code in plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False

# Common setup for all plot directives
plot_pre_code = '''
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax_supernovae import SALT3Source, TimeSeriesSource
from jax_supernovae.bandpasses import get_bandpass, Bandpass, register_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge
'''

# Figure format for plots
plot_formats = ['png']