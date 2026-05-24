assistant — human-in-the-loop inbox
=====================================

In-process inbox used by the HITL middleware to pause a ReAct turn at a
Genie-synthesize call, surface a preview to the user, and wait briefly
for an ``approve`` / ``skip`` / ``modify`` decision before either
continuing or aborting the dispatch.

.. note::
    The inbox is process-local. Multi-worker deployments would need a
    Redis-backed swap so the inbox is shared across replicas — currently
    the SSE listener and the inbox writer must run in the same process.

.. automodule:: app.assistant.hitl_inbox
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
