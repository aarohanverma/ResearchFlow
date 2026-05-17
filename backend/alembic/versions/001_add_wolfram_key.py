"""Add encrypted_wolfram_key to user_provider_settings

Revision ID: 001_add_wolfram_key
Revises:
Create Date: 2026-05-15
"""

from alembic import op
import sqlalchemy as sa

revision = "001_add_wolfram_key"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_provider_settings",
        sa.Column("encrypted_wolfram_key", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_provider_settings", "encrypted_wolfram_key")
