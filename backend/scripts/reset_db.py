"""Reset script — wipes all ingested data and caches, preserves the test user.

What is cleared:
  - All papers, chunks, summaries, bookmarks, feed feedback, citations, paper-of-day
  - All knowledge graph nodes and edges
  - All workflow run logs and token usage records
  - All Genie elements, idea capsules, and sessions
  - All RAG query logs and annotations
  - User interest profile subtopics (hot/cold reset to empty lists)
  - Local cache directory (file cache)
  - Local blob directory (stored PDFs, images)
  - Redis cache (if Redis is reachable)

What is preserved:
  - Test user account and password
  - User provider settings
  - User interest profile row (kept, subtopics emptied)
  - Namespace subscriptions (cs.AI, cs.ML, cs.NLP)
  - Source mappings (so Refresh Feed works immediately)

Run:
  cd backend && python scripts/reset_db.py
"""

import asyncio
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.session import async_session_factory, create_all_tables
from app.core.config import settings

# ── Colours ───────────────────────────────────────────────────────────────────
RED    = "\033[0;31m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
RESET  = "\033[0m"

def ok(msg):  print(f"{GREEN}✓  {msg}{RESET}")
def info(msg):print(f"{CYAN}ℹ  {msg}{RESET}")
def warn(msg):print(f"{YELLOW}⚠  {msg}{RESET}")
def err(msg): print(f"{RED}✗  {msg}{RESET}")


# ── Tables to truncate in dependency order ────────────────────────────────────
# Must go from most-dependent → least-dependent so FK constraints are satisfied.
# Tables with ondelete="CASCADE" at the DB level are handled automatically when
# their parent is deleted, but we truncate them explicitly for clarity and speed.
TRUNCATE_TABLES = [
    # Genie (user FK — user kept, so must delete manually)
    "genie_sessions",
    "idea_capsules",
    "genie_elements",

    # Workflow + observability (no user FK or nullable)
    "token_usage",
    "workflow_runs",

    # RAG
    "query_logs",

    # Graph (edge → node, CASCADE at DB; node → paper, nullable)
    "knowledge_edges",
    "knowledge_nodes",

    # Papers and everything that cascades from them:
    # paper_chunks, summaries, bookmarks, feed_feedback,
    # paper_citations, paper_of_day  — all cascade at DB level
    "papers",
]

# Tables we partially update (reset content, keep row)
RESET_PROFILE_SQL = """
    UPDATE user_interest_profiles
    SET hot_subtopics = '{}',
        cold_subtopics = '{}',
        updated_at = NOW()
    WHERE true
"""

# Also clear annotations if the table exists
ANNOTATION_CLEAR_SQL = "DELETE FROM annotations"


async def reset_database() -> None:
    info("Connecting to database …")
    await create_all_tables()  # Ensure tables exist (idempotent)

    async with async_session_factory() as db:
        # Confirm user exists before wiping anything
        result = await db.execute(text("SELECT email FROM users LIMIT 1"))
        row = result.fetchone()
        if not row:
            warn("No users found — nothing to preserve. Run seed_db.py first.")
            return

        info(f"Preserving user: {row[0]}")
        print()

        # ── Truncate data tables ──────────────────────────────────────────────
        info("Clearing data tables …")
        for table in TRUNCATE_TABLES:
            try:
                result = await db.execute(text(f"DELETE FROM {table}"))
                count = result.rowcount
                ok(f"  {table:<30} {count:>6} rows removed")
            except Exception as e:
                warn(f"  {table:<30} skipped ({e})")

        # ── Clear annotations (may not exist yet) ─────────────────────────────
        try:
            result = await db.execute(text(ANNOTATION_CLEAR_SQL))
            ok(f"  {'annotations':<30} {result.rowcount:>6} rows removed")
        except Exception:
            pass  # table may not exist in all migrations

        # ── Reset interest profile ────────────────────────────────────────────
        try:
            await db.execute(text(RESET_PROFILE_SQL))
            ok(f"  {'user_interest_profiles':<30} subtopics reset")
        except Exception as e:
            warn(f"  user_interest_profiles: {e}")

        await db.commit()

    print()
    ok("Database reset complete.")


def clear_filesystem() -> None:
    info("Clearing local filesystem caches …")

    # Cache directory
    cache_dir = settings.cache_dir
    if os.path.isdir(cache_dir):
        cleared = 0
        for entry in os.scandir(cache_dir):
            # Don't delete the blobs subdirectory itself — cleared separately
            if entry.is_dir() and entry.name == "blobs":
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)
                cleared += 1
            except Exception as e:
                warn(f"  Could not remove {entry.path}: {e}")
        ok(f"  {cache_dir}: {cleared} entries removed")
    else:
        info(f"  Cache dir not found ({cache_dir}) — skipping")

    # Blob directory
    blob_dir = settings.blob_local_dir
    if os.path.isdir(blob_dir):
        cleared = 0
        for entry in os.scandir(blob_dir):
            try:
                if entry.is_dir():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)
                cleared += 1
            except Exception as e:
                warn(f"  Could not remove {entry.path}: {e}")
        ok(f"  {blob_dir}: {cleared} blobs removed")
    else:
        info(f"  Blob dir not found ({blob_dir}) — skipping")


def clear_redis() -> None:
    if settings.cache_backend != "redis":
        info("Redis cache backend not active — skipping Redis flush")
        return

    info("Flushing Redis cache …")
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=3)
        r.flushdb()
        ok(f"  Redis db at {settings.redis_url} flushed")
    except Exception as e:
        warn(f"  Redis not reachable or flush failed: {e}")
        warn("  (Redis is optional — local file cache was cleared above)")


async def main() -> None:
    print()
    print(f"{CYAN}{'━' * 60}{RESET}")
    print(f"{CYAN}  ResearchFlow — App State Reset{RESET}")
    print(f"{CYAN}{'━' * 60}{RESET}")
    print()

    # Confirm before destructive action
    print(f"  This will delete ALL papers, graph, genie, and workflow data.")
    print(f"  The test user account will be {GREEN}preserved{RESET}.")
    print()
    confirm = input("  Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        info("Aborted — nothing was changed.")
        return

    print()

    await reset_database()
    print()
    clear_filesystem()
    clear_redis()

    print()
    print(f"{CYAN}{'━' * 60}{RESET}")
    print(f"{GREEN}  Reset complete.{RESET}")
    print()
    print("  Next steps:")
    print("    1. Go to Settings → Refresh Feed Manually → pick a namespace → Refresh Now")
    print("    2. Wait 30–60 seconds for papers to appear")
    print("    3. Or: POST /api/v1/feed/refresh?namespace_key=cs.AI")
    print()
    print(f"  Login credentials unchanged:")
    print(f"    Email:    {GREEN}test@researchflow.ai{RESET}")
    print(f"    Password: {GREEN}ResearchFlow2024!{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
