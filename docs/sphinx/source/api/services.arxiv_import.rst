services — arXiv import
=========================

Authoritative arXiv search-and-import service used by both the
``/assistant/arxiv/*`` endpoints and the Research Assistant
``arxiv_import`` tool. Wraps a configurable arXiv adapter
(``arxiv_rss`` or the official ``arxiv-mcp-server``) and runs the full
enrichment + embedding + graph-update pipeline for each imported paper.

.. automodule:: app.services.arxiv_import
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
