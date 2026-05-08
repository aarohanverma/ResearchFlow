"""Sphinx configuration for ResearchFlow backend documentation."""

import os
import sys

# Make the backend package importable without installing it
# backend/ must be on sys.path so `import app` resolves
_backend = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "backend")
)
sys.path.insert(0, _backend)

# ── Project info ──────────────────────────────────────────────────────────────
project = "ResearchFlow"
copyright = "2026, Aarohan Verma"
author = "ResearchFlow Team"
release = "1.0.0"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",         # pulls docstrings from source
    "sphinx.ext.napoleon",        # Google + NumPy style docstrings
    "sphinx.ext.viewcode",        # [source] links next to each function
    "sphinx.ext.intersphinx",     # cross-links to Python stdlib docs
    "sphinx.ext.autosummary",     # auto-generate summary tables
    "sphinx.ext.githubpages",
]

# ── Napoleon (Google-style docstrings) ───────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# ── Autodoc ───────────────────────────────────────────────────────────────────
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__init__",
    "show-inheritance": True,
    "inherited-members": False,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "separated"
add_module_names = False

# ── Autosummary ───────────────────────────────────────────────────────────────
autosummary_generate = True

# ── Intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
}

# ── HTML theme ────────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ["_static"]
html_title = "ResearchFlow — API Reference"
html_short_title = "ResearchFlow"

# ── Mock unavailable heavy imports ───────────────────────────────────────────
# These packages may not be installed in the Sphinx build environment
autodoc_mock_imports = [
    "openai", "anthropic", "google", "google.generativeai",
    "pgvector", "pgvector.sqlalchemy",
    "sqlalchemy", "alembic",
    "fastapi", "pydantic", "pydantic_settings",
    "langchain", "langgraph",
    "bcrypt", "jose",
    "aiohttp", "httpx", "aiofiles",
    "redis", "celery",
    "marker", "marker_pdf",
    "azure", "azure.storage",
    "resend",
    "apscheduler",
    "apscheduler.schedulers",
    "apscheduler.schedulers.asyncio",
    "numpy",
]

# Suppress known false-positive warnings
suppress_warnings = [
    "autodoc.import_object",  # scheduler/jobs.py APScheduler mock type issue
]

# ── Misc ──────────────────────────────────────────────────────────────────────
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "monokai"
