"""Seed script — creates a test user and initial SourceMappings.

Run once after `docker-compose up` and DB migration:
  cd backend && python scripts/seed_db.py

Test user credentials:
  Email:    test@researchflow.ai
  Password: ResearchFlow2024!
"""

import asyncio
import sys
import os

# Ensure backend/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from app.db.session import async_session_factory, create_all_tables
from app.core.security import hash_password
from app.models.user import User, UserProviderSettings, UserInterestProfile, ExpertiseLevel, Orientation
from app.models.graph import NamespaceSubscription, SourceMapping

TEST_EMAIL = "test@researchflow.ai"
TEST_PASSWORD = "ResearchFlow2024!"
TEST_DISPLAY_NAME = "Test Researcher"

# Default subscriptions for the test user
DEFAULT_NAMESPACES = [
    ("cs.AI",     "arxiv_rss", "cs.AI"),
    ("cs.ML",     "arxiv_rss", "cs.LG"),
    ("cs.NLP",    "arxiv_rss", "cs.CL"),
]


async def seed():
    print("Creating tables...")
    await create_all_tables()

    async with async_session_factory() as db:
        # ── Test User ─────────────────────────────────────────────────────────
        existing = await db.execute(select(User).where(User.email == TEST_EMAIL))
        user = existing.scalar_one_or_none()

        if user:
            print(f"Test user already exists: {TEST_EMAIL}")
        else:
            user = User(
                email=TEST_EMAIL,
                hashed_password=hash_password(TEST_PASSWORD),
                display_name=TEST_DISPLAY_NAME,
                expertise_level=ExpertiseLevel.practitioner,
                orientation=Orientation.both,
                notify_potd=True,
                notify_digest=True,
                notify_breakthrough=True,
                onboarding_complete=True,
            )
            db.add(user)
            await db.flush()

            # Provider settings (defaults)
            db.add(UserProviderSettings(user_id=user.id))

            # Empty interest profile
            db.add(UserInterestProfile(user_id=user.id))

            print(f"✓ Created test user: {TEST_EMAIL}")

        # ── Namespace subscriptions ────────────────────────────────────────────
        for ns_key, source_name, arxiv_cat in DEFAULT_NAMESPACES:
            # Subscription
            sub_exists = await db.execute(
                select(NamespaceSubscription).where(
                    NamespaceSubscription.user_id == user.id,
                    NamespaceSubscription.namespace_key == ns_key,
                )
            )
            if not sub_exists.scalar_one_or_none():
                db.add(NamespaceSubscription(user_id=user.id, namespace_key=ns_key))
                print(f"  + Subscribed to {ns_key}")

            # SourceMapping
            mapping_exists = await db.execute(
                select(SourceMapping).where(
                    SourceMapping.namespace_key == ns_key,
                    SourceMapping.source_name == source_name,
                )
            )
            if not mapping_exists.scalar_one_or_none():
                db.add(SourceMapping(
                    namespace_key=ns_key,
                    source_name=source_name,
                    external_category_key=arxiv_cat,
                ))
                print(f"  + SourceMapping {ns_key} → arXiv {arxiv_cat}")

        await db.commit()

    print("\n✓ Seed complete.")
    print(f"\n  Login credentials:")
    print(f"    Email:    {TEST_EMAIL}")
    print(f"    Password: {TEST_PASSWORD}")
    print(f"\n  To fetch papers immediately:")
    print(f"    POST /api/v1/feed/refresh?namespace_key=cs.AI")
    print(f"    (or use the Settings page → Refresh Feed Manually)")


if __name__ == "__main__":
    asyncio.run(seed())
