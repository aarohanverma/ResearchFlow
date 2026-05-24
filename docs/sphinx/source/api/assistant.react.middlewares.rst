assistant — ReAct middleware chain
====================================

Concrete middlewares composed by ``default_chain_factory`` and walked by
the ReAct loop's ``MiddlewareChain``. Each middleware is independently
testable; cross-cutting concerns (param hygiene, tool bans, HITL gating,
redundancy detection, paper-ID accumulation, retrieval observability,
critique gating, contradiction detection, full-paper verification) live
in their own file.

.. automodule:: app.assistant.react.middlewares
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.param_preflight
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.tool_ban
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.hitl_gate
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.diminishing_returns
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.paper_ledger
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.observability_mw
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.critic_gate
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.contradiction_mw
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: app.assistant.react.middlewares.full_paper_gate
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
