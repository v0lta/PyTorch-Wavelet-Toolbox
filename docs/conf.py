# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------

project = "PyTorch-Wavelet-Toolbox"
copyright = "2025"
author = "Moritz Wolter, Felix Blanke, Jochen Garcke and Charles Tapley Hoyt"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

napoleon_google_docstring = True
# napoleon_use_admonition_for_examples = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# document __init__ in the docpages
autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/v0lta/PyTorch-Wavelet-Toolbox",
    "use_repository_button": True,
    "navigation_with_keys": False,
}

html_favicon = "_static/favicon.ico"
html_logo = "_static/shannon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


bibtex_bibfiles = ["literature.bib"]

# numbered figures
numfig = True

autodoc_type_aliases = {
    "WaveletCoeff2d": "ptwt.constants.WaveletCoeff2d",
    "WaveletCoeff2dSeparable": "ptwt.constants.WaveletCoeff2dSeparable",
    "WaveletCoeffNd": "ptwt.constants.WaveletCoeffNd",
    "BaseMatrixWaveDec": "ptwt.matmul_transform.BaseMatrixWaveDec",
    "BoundaryMode": "ptwt.constants.BoundaryMode",
    "ExtendedBoundaryMode": "ptwt.constants.ExtendedBoundaryMode",
    "OrthogonalizeMethod": "ptwt.constants.OrthogonalizeMethod",
}
