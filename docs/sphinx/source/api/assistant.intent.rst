assistant — intent + complexity classification
================================================

Heuristic intent tier classifier that decides whether the orchestrator
takes a fast direct-reply path, a single-tool path, or the full ReAct
loop. Also produces complexity hints the planner uses when picking
between cheap / quality / reasoning models.

.. automodule:: app.assistant.intent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
