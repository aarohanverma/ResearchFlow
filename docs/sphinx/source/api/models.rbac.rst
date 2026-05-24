models — RBAC scaffolding
===========================

Forward-compatible role and tier columns on ``User`` (``role`` string,
``tier_slug`` for future subscription tiers). Today every authorisation
decision still reads ``users.is_admin``; this module exists so adding
editor / reviewer / paying-tier roles in the future does not require
another migration.

.. automodule:: app.models.rbac
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
