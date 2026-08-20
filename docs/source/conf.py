"""Sphinx configuration for NeuRelux, built by ReadTheDocs (see .readthedocs.yaml).

Deliberately dependency-light: API docs use sphinx-autoapi, which parses
src/atlas_physics statically instead of importing it — so the docs build does
NOT need torch (or any other runtime dependency) installed. This keeps the
ReadTheDocs build fast and decoupled from the notebook/training environment.
Notebooks are rendered from their already-executed .ipynb outputs
(nbsphinx_execute = "never") rather than re-run on the docs server — see
notebooks/01_skin_effect_cauer_synthetic.ipynb, which takes ~2 minutes to
train and should not be retrained on every docs build.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "NeuRelux"
author = "NeuRelux contributors"
copyright = "2026, NeuRelux contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "autoapi.extension",
    "sphinxcontrib.bibtex",
]

# -- MyST / source parsing --------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

# -- nbsphinx -----------------------------------------------------------------
nbsphinx_execute = "never"

# -- autoapi (static analysis of src/atlas_physics, no import required) ------
autoapi_type = "python"
autoapi_dirs = ["../../src/atlas_physics"]
autoapi_add_toctree_entry = True
autoapi_keep_files = False

# -- bibliography --------------------------------------------------------------
bibtex_bibfiles = ["../../papers/references.bib"]
bibtex_default_style = "plain"

# -- HTML output ---------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "NeuRelux"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# README.md's relative link to PLAN.md (a plain GitHub-relative link, correct
# on GitHub) isn't a resolvable Sphinx doc target once included verbatim into
# readme.md here — cosmetic only, the real cross-navigation is the toctree above.
suppress_warnings = ["myst.xref_missing"]
