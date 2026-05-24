assistant — memory consolidation
==================================

Weekly background pass that clusters and LLM-merges related session-
memory entries across every user's chat / tree / namespace tiers. Keeps
memory bounded by *summary* rather than raw eviction. Scheduled by
``app.scheduler.jobs`` on Sundays at 04:30 UTC.

.. automodule:: app.assistant.memory_consolidation
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
