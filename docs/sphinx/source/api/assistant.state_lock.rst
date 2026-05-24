assistant — state lock
========================

Async per-session lock used to serialise writes to ``AssistantSession.state``
so concurrent turns / branches / consolidators never clobber each other's
JSONB patches.

.. automodule:: app.assistant.state_lock
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
