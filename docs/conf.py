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


sys.path.insert(1, os.path.abspath(os.getcwd()+"/.."))
sys.path.insert(1, os.path.abspath(os.getcwd()+"/../cython_filter_new/"))

sys.path.insert(1, os.path.abspath('../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../ManipulationLibrary/'))

# sys.path.insert(1, os.path.dirname(os.path.abspath("../")) + os.sep + "feature_engine")

# from cython_filter_new import filter_cython_new

# print(sys.path)
# print(os.path.abspath(os.getcwd()))

# if 'READTHEDOCS' not in os.environ:
#     from cython_filter_new import filter_cython_new

autodoc_mock_imports = ["tabula", "pandas", "numpy", "matplotlib", "scipy", "itertools", "seaborn", "open3d", "cv2", "progressbar", "sklearn", "cython_filter_new/filter_cython_new", "filter_cython_new"]


# -- Project information -----------------------------------------------------

project = 'ManipulationLibrary'
copyright = '2020, Tobias Strübing'
author = 'Tobias Strübing'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
master_doc = 'index'

