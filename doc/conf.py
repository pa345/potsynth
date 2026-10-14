# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'potsynth'
copyright = '2026, Patrick Alken and Jonathan Burchill'
author = 'Patrick Alken and Jonathan Burchill'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# standard C types are not documented here
nitpick_ignore = [('c:identifier', 'size_t')]

numfig = True
math_numfig = True
math_eqref_format = 'Eq. {number}'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = []
