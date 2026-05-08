"""SQLAlchemy declarative base — imported by every model."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base shared by all ORM models.

    Import this class in every model module and use it as the base class
    so Alembic can discover all table definitions via ``Base.metadata``.
    """

    pass


# Re-export so Alembic env.py can do: from app.db.base import Base
__all__ = ["Base"]
