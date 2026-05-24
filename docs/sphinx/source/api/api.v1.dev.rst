Developer router
================

Dev-only endpoints, gated by ``ENABLE_DEV_RESET=true``. The router is
always mounted but every action returns ``404 Not Found`` unless the env
flag is set, so it is safe to include in production builds.

.. automodule:: app.api.v1.dev
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
