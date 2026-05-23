"""Sphinx configuration for the flatwalk documentation."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

# Make the in-tree examples/ importable so example code in the docs can
# reference it (e.g. for autodoc on examples.ising).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


# -- Project information ----------------------------------------------------

project = "flatwalk"
author = "Henrik Jaeger"
copyright = "2026, Henrik Jaeger"

try:
    release = importlib.metadata.version("flatwalk")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])


# -- General configuration --------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]


# -- Sphinx Gallery ---------------------------------------------------------

sphinx_gallery_conf = {
    # Two galleries, both with sources next to docs/src/ (separate from the
    # project's examples/ directory, which holds the CLI/validation scripts):
    #   docs/examples/  -> auto_examples/   recipes (write-your-own-contract)
    #   docs/tutorials/ -> auto_tutorials/  a narrative journey on one system
    "examples_dirs": [
        str(ROOT / "docs" / "examples"),
        str(ROOT / "docs" / "tutorials"),
    ],
    "gallery_dirs": ["auto_examples", "auto_tutorials"],  # written below docs/src/
    "filename_pattern": r"plot_",
    # The examples form a tutorial sequence; order by filename (the numeric
    # plot_N_ prefixes) rather than the default code-length sort. The string
    # form keeps sphinx_gallery_conf picklable (no class object in config).
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    "doc_module": ("flatwalk",),
    # Generate per-object backreference stubs so the API page can surface, via
    # `.. minigallery::`, which examples/tutorials exercise each flatwalk object.
    "backreferences_dir": "gen_modules/backreferences",
    "reference_url": {"flatwalk": None},
    "default_thumb_file": None,
    "abort_on_example_error": True,
    "min_reported_time": 1,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Cross-references to external docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST (markdown) extensions. dollarmath + amsmath let the math-heavy theory
# pages use $…$ / $$…$$, rendered by the mathjax extension above.
myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "smartquotes",
    "tasklist",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

# Autodoc behaviour: order members as they appear in source, render type
# hints in the description (Furo handles them cleanly), pick up class
# attributes from `__init__`.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
# Render NumPy-style Attributes sections as :ivar:; avoids autodoc
# emitting duplicate entries when the same attribute appears in both the
# dataclass field list and the docstring's Attributes section.
napoleon_use_ivar = True


# -- HTML output ------------------------------------------------------------

html_theme = "furo"
html_title = f"flatwalk {version}"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}
