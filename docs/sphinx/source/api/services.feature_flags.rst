services — feature flags
==========================

Per-user and global feature-flag plumbing. Backed by JSONB on the
``users.feature_overrides`` column for per-user overrides plus the
admin-managed ``app_settings`` table for global defaults.

.. automodule:: app.services.feature_flags
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
