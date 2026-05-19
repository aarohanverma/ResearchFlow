"""Application-wide settings managed by admins.

Currently exposes a single ``AppSetting`` table for feature flags and admin
configuration. Today it holds one canonical row keyed by ``"global"`` whose
JSON value is whatever flag set the running app needs (graph_enabled, …);
storing it as JSONB keeps it cheap to evolve without per-flag migrations.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class AppSetting(Base):
    """Key/JSON setting row. The app reads the row with ``key='global'``."""

    __tablename__ = "app_settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    value: Mapped[dict] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
