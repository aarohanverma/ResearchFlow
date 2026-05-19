"""Single source of truth for per-feature flags and their resolution.

Three layers, last-wins precedence:

  1. ``DEFAULTS``   — hardcoded default for every feature.
  2. Global admin   — ``AppSetting(key='global').value`` (set by admin panel).
  3. Per-user      — ``users.feature_overrides`` (set per-user by admin).

A feature is *enabled for user X* iff the merged dict, in that order, has
``True`` (or any truthy value) for the feature key.

Everything routed through ``is_feature_enabled`` / ``get_effective_features``
so call-sites never read the raw dicts directly.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory
from app.models.user import User
from app.services.admin_settings import get_app_settings

log = logging.getLogger(__name__)


# ── Canonical feature set ─────────────────────────────────────────────────────
# Every feature lives here with a docstring + default. The frontend reads
# this catalog (via ``/settings/features``) to render the admin panel
# fine-grained toggles, so adding a flag here automatically makes it
# admin-toggleable.

FEATURES: dict[str, dict[str, Any]] = {
    # Knowledge Graph (build, view, expand, all related routes). Default off
    # because the current builder doesn't scale well — admins flip it on
    # explicitly. The flag also fully disconnects RA tools that depend on it.
    "graph_enabled": {
        "default": False,
        "label": "Knowledge Graph",
        "description": "Build / browse / search the cross-paper concept graph.",
    },
    # Genie idea synthesis (manual, auto, query, combine). Off disables all
    # three modes consistently — UI hides Genie nav and tools short-circuit.
    "genie_enabled": {
        "default": True,
        "label": "Genie (idea synthesis)",
        "description": "Synthesize, combine, and explore research ideas from saved papers.",
    },
    "genie_auto_enabled": {
        "default": True,
        "label": "Genie · Auto-discovery",
        "description": "Automatic clustering + idea synthesis on the user's bookmarks.",
    },
    "genie_combine_enabled": {
        "default": True,
        "label": "Genie · Combine",
        "description": "Fuse 2–3 existing ideas into a hybrid hypothesis.",
    },
    # Research Assistant chat — the conversational workspace.
    "assistant_enabled": {
        "default": True,
        "label": "Research Assistant",
        "description": "Conversational research workspace with tools, memory, and citations.",
    },
    # Deep / hybrid search on the Feed.
    "deep_search_enabled": {
        "default": True,
        "label": "Deep Search",
        "description": "LLM-rewritten hybrid retrieval with arXiv top-up on the Feed.",
    },
    # arXiv ingestion / nightly feed jobs.
    "arxiv_ingest_enabled": {
        "default": True,
        "label": "arXiv ingestion",
        "description": "Nightly arXiv ingestion and on-demand MCP imports.",
    },
    # Paper study mode (audio, slides, deep walkthrough).
    "study_mode_enabled": {
        "default": True,
        "label": "Study Mode",
        "description": "Full paper walkthroughs, audio narration, slide generation.",
    },
}

DEFAULTS: dict[str, bool] = {k: bool(v["default"]) for k, v in FEATURES.items()}


# ── Resolution helpers ────────────────────────────────────────────────────────


async def get_effective_features(
    user_id: UUID | None,
    db: AsyncSession | None = None,
) -> dict[str, bool]:
    """Return the merged feature dict for ``user_id``.

    Layers, last wins:
        1. DEFAULTS               (compile-time defaults)
        2. global admin settings  (admin panel toggles)
        3. tier feature_set       (user's subscription tier, if any)
        4. per-user overrides     (admin's per-user force-on/off)

    Boolean coercion is applied at every layer so JSONB ``None`` / strings
    never accidentally enable a gated feature.
    """
    merged: dict[str, bool] = dict(DEFAULTS)

    # Layer 2 — global flags set via the admin panel.
    try:
        global_settings = await get_app_settings(db)
        for k in DEFAULTS:
            if k in global_settings:
                merged[k] = bool(global_settings[k])
    except Exception as exc:
        log.debug("get_effective_features: global settings unavailable, using defaults (%s)", exc)

    if user_id is None:
        return merged

    # Resolve user + (optional) tier in a single trip. Falls back gracefully
    # if either query fails — never blocks the request on RBAC plumbing.
    async def _fetch_user_and_tier(session: AsyncSession) -> tuple[dict[str, Any], dict[str, Any]]:
        from app.models.rbac import Tier  # local import to keep RBAC optional

        u_row = await session.execute(select(User).where(User.id == user_id))
        user = u_row.scalar_one_or_none()
        if user is None:
            return {}, {}
        overrides = dict(getattr(user, "feature_overrides", {}) or {})
        tier_features: dict[str, Any] = {}
        tier_slug = getattr(user, "tier_slug", None)
        if tier_slug:
            t_row = await session.execute(select(Tier).where(Tier.slug == tier_slug))
            tier = t_row.scalar_one_or_none()
            if tier is not None:
                tier_features = dict(getattr(tier, "feature_set", {}) or {})
        return tier_features, overrides

    try:
        if db is not None:
            tier_features, overrides = await _fetch_user_and_tier(db)
        else:
            async with async_session_factory() as session:
                tier_features, overrides = await _fetch_user_and_tier(session)
    except Exception as exc:
        log.debug("get_effective_features: tier / override fetch failed (%s)", exc)
        tier_features, overrides = {}, {}

    # Layer 3 — tier's feature_set (subscription package).
    for k, v in tier_features.items():
        if k in DEFAULTS:
            merged[k] = bool(v)
    # Layer 4 — admin's per-user override (final say).
    for k, v in overrides.items():
        if k in DEFAULTS:
            merged[k] = bool(v)
    return merged


async def is_feature_enabled(
    feature: str,
    user_id: UUID | None,
    db: AsyncSession | None = None,
) -> bool:
    """Return True iff ``feature`` is enabled for ``user_id``.

    Unknown features default to ``False`` to fail safely — typo in a call
    site shouldn't accidentally unlock a gated path.
    """
    if feature not in DEFAULTS:
        return False
    eff = await get_effective_features(user_id, db)
    return bool(eff.get(feature, DEFAULTS[feature]))


async def is_global_feature_enabled(feature: str) -> bool:
    """Same as ``is_feature_enabled`` but ignores any user — checks the
    global (admin-controlled) value only.

    Useful for system-level call sites (nightly ingestion jobs, scheduled
    tasks, RAG retrieval expansions) where there is no single "owner"
    user to resolve overrides against. Per-user overrides apply only at
    user-facing call sites.
    """
    if feature not in DEFAULTS:
        return False
    try:
        s = await get_app_settings()
    except Exception:
        return bool(DEFAULTS[feature])
    return bool(s.get(feature, DEFAULTS[feature]))


async def set_user_overrides(user_id: UUID, patch: dict[str, Any]) -> dict[str, bool]:
    """Update a user's ``feature_overrides`` map and return the new effective set."""
    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise LookupError("User not found")
        current = dict(user.feature_overrides or {})
        for k, v in patch.items():
            if k not in DEFAULTS:
                continue
            if v is None:
                current.pop(k, None)  # null clears the override → inherit global
            else:
                current[k] = bool(v)
        user.feature_overrides = current
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(user, "feature_overrides")
        await session.commit()
    return await get_effective_features(user_id)
