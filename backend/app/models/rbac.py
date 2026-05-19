"""RBAC scaffolding — Tier table for future subscription tiers.

Two ideas are kept deliberately separate:

* **Role** — tiny enum (``admin`` | ``member``) that controls access to
  management surfaces. Encoded today as ``users.is_admin``; the string
  ``users.role`` column added in this migration is forward-compatible
  with a future role table without changing app code.

* **Tier** — product packaging. Each ``Tier`` row carries:

  * ``feature_set``: JSONB map of feature_flag → bool, layered between
    "global admin settings" and "per-user override" in the resolution
    chain (see :mod:`app.services.feature_flags`).
  * ``quotas``: JSONB of soft caps the app can enforce later (max ideas
    per month, max imports per day, etc.). Not enforced today; the
    shape is reserved so quota logic can be added without a migration.

Users opt into a tier by setting ``users.tier_slug``. ``None`` means the
user inherits the global flag values directly — which is the only state
that exists today, so this whole module is a no-op until an admin
creates tiers and assigns users to them.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class Tier(Base):
    """Subscription tier — defines a default feature_set + quotas."""

    __tablename__ = "tiers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(40), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    # Map of feature_key → bool. Keys must come from
    # ``app.services.feature_flags.DEFAULTS``; unknown keys are ignored
    # at resolution time so a stale flag never crashes the request.
    feature_set: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Map of quota_key → int (or other primitive). Reserved for the
    # quota-enforcement layer — empty by default and unused today.
    quotas: Mapped[dict] = mapped_column(JSONB, default=dict)
    # When True, this tier is the catch-all "no subscription / free" tier.
    # New users default to it iff one is marked default; if none is the
    # default, ``users.tier_slug`` stays NULL and global flags apply.
    is_default: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
