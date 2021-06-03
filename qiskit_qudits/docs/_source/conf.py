# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import qiskit_sphinx_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath('../../..'))

# -- Project information -----------------------------------------------------

project = 'qiskit-qudits'
author = 'Hoang Van Do, Tim Alexis Körner, Inho Choi, Timothé Presles and Élie Gouzien'
copyright = 'Apache 2.0, 2021 ' + author

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx_automodapi.automodapi',
    'jupyter_sphinx',
    'nbsphinx',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'colorful'

# A boolean that decides whether module names are prepended to all object names
# (for object types where a “module” of some kind is defined), e.g. for
# py:function directives.
add_module_names = False

# A list of prefixes that are ignored for sorting the Python module index
# (e.g., if this is set to ['foo.'], then foo.bar is shown under B, not F).
# This can be handy if you document a project that consists of a single
# package. Works only for the HTML builder currently.
modindex_common_prefix = ['qiskit.']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "qiskit_sphinx_theme"

html_theme_path = ['.', qiskit_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_last_updated_fmt = '%Y/%m/%d'

autosummary_generate = True
autosummary_generate_overwrite = False

autodoc_default_options = {
    'inherited-members': None,
}

autoclass_content = 'both'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['*test*']
