assistant — strong-claim ledger
================================

Detects "strong" claims (numeric/SOTA/causal/comparative spans) in tool
results, tags each with its source provenance (``SOURCE_CHUNK`` from
``paper_qa`` is verified; ``SOURCE_ABSTRACT`` / ``SOURCE_SNIPPET`` from
retrieval is provisional), and feeds the result to the full-paper
verification middleware.

.. automodule:: app.assistant.claim_ledger
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
