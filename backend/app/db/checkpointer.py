"""Async PostgreSQL checkpoint saver for LangGraph 0.2.x / langgraph-checkpoint 2.x.

Implements BaseCheckpointSaver using asyncpg directly — no external
langgraph-checkpoint-postgres package required (avoids the langgraph 0.2 ↔
checkpoint-postgres 3.x version clash).

Schema: three tables (checkpoints, blobs, writes) created idempotently on
startup.  Each table is keyed by (thread_id, checkpoint_ns).  The thread_id
for generation workflows is always the artifact UUID, so checkpoints survive
server restarts and the workflow can resume from the last completed node
instead of regenerating from scratch.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Iterator
from typing import Any, Sequence

import asyncpg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

log = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
        thread_id           TEXT NOT NULL,
        checkpoint_ns       TEXT NOT NULL DEFAULT '',
        checkpoint_id       TEXT NOT NULL,
        parent_checkpoint_id TEXT,
        type                TEXT NOT NULL,
        checkpoint          BYTEA NOT NULL,
        metadata_type       TEXT NOT NULL DEFAULT 'json',
        metadata            BYTEA NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS langgraph_checkpoint_blobs (
        thread_id     TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        channel       TEXT NOT NULL,
        version       TEXT NOT NULL,
        type          TEXT NOT NULL,
        blob          BYTEA,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS langgraph_checkpoint_writes (
        thread_id     TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        task_id       TEXT NOT NULL,
        idx           INTEGER NOT NULL,
        channel       TEXT NOT NULL,
        type          TEXT,
        blob          BYTEA NOT NULL,
        task_path     TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )
    """,
]


# ── Checkpointer ──────────────────────────────────────────────────────────────


class AsyncPostgresCheckpointer(BaseCheckpointSaver):
    """PostgreSQL-backed LangGraph checkpoint saver (asyncpg).

    Checkpoints are stored durably so interrupted workflows resume from the
    last completed LangGraph node instead of restarting from scratch.

    Usage::

        checkpointer = await AsyncPostgresCheckpointer.create(dsn)
        graph = builder.compile(checkpointer=checkpointer)
        await graph.ainvoke(state, config={"configurable": {"thread_id": artifact_id}})

    The ``thread_id`` should be the artifact UUID so each generation job has
    its own isolated checkpoint namespace.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialise the checkpointer with an existing asyncpg connection pool.

        Prefer :meth:`create` which creates the pool and runs schema setup
        atomically. Use this constructor only when you already have a pool.

        Args:
            pool: An asyncpg connection pool connected to the target database.
        """
        super().__init__()
        self._pool = pool

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, dsn: str) -> "AsyncPostgresCheckpointer":
        """Create a pool and ensure checkpoint tables exist.

        Args:
            dsn: asyncpg-compatible PostgreSQL DSN (``postgresql://...``).

        Returns:
            Ready-to-use :class:`AsyncPostgresCheckpointer`.
        """
        pool = await asyncpg.create_pool(
            dsn, min_size=1, max_size=4,
            command_timeout=30,
        )
        inst = cls(pool)
        await inst._setup()
        return inst

    async def _setup(self) -> None:
        """Create the three LangGraph checkpoint tables idempotently.

        Uses ``CREATE TABLE IF NOT EXISTS`` so repeated calls are safe.
        Runs all DDL statements in a single transaction.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for stmt in _DDL:
                    await conn.execute(stmt)

    async def close(self) -> None:
        """Close the underlying asyncpg connection pool.

        Should be called during application shutdown to release all DB
        connections cleanly. Idempotent — safe to call on an already-closed
        pool.
        """
        await self._pool.close()

    # ── Blob helpers ──────────────────────────────────────────────────────────

    async def _load_blobs(
        self,
        conn: asyncpg.Connection,
        thread_id: str,
        checkpoint_ns: str,
        versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Fetch channel values for a set of (channel → version) mappings.

        Issues a single batch SELECT instead of one query per channel so
        checkpoint reads scale O(1) in DB round-trips regardless of how many
        channels a workflow has.

        Args:
            conn: Active asyncpg connection.
            thread_id: LangGraph thread identifier (artifact UUID).
            checkpoint_ns: Checkpoint namespace (empty string for default).
            versions: Dict mapping channel name to version string.

        Returns:
            Dict mapping channel name to deserialised channel value for all
            non-empty blobs found in the table.
        """
        channel_values: dict[str, Any] = {}
        if not versions:
            return channel_values

        # Single batch query: fetch all channels for this thread at once.
        # WHERE channel = ANY(…) AND version = ANY(…) over-selects when two
        # channels share the same version string, but we verify exact
        # (channel, version) pairs below so correctness is preserved.
        channels = list(versions.keys())
        vers = [str(v) for v in versions.values()]
        rows = await conn.fetch(
            "SELECT channel, version, type, blob "
            "FROM langgraph_checkpoint_blobs "
            "WHERE thread_id=$1 AND checkpoint_ns=$2 "
            "  AND channel = ANY($3::text[]) "
            "  AND version = ANY($4::text[])",
            thread_id, checkpoint_ns, channels, vers,
        )

        # Build a (channel, version) → row lookup for exact-pair matching
        row_map: dict[tuple[str, str], Any] = {
            (r["channel"], r["version"]): r for r in rows
        }
        for channel, version in versions.items():
            row = row_map.get((channel, str(version)))
            if row and row["type"] != "empty" and row["blob"] is not None:
                channel_values[channel] = self.serde.loads_typed(
                    (row["type"], bytes(row["blob"]))
                )
        return channel_values

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Synchronous checkpoint lookup — not supported; always raises.

        Raises:
            NotImplementedError: Always. Use :meth:`aget_tuple` instead.
        """
        raise NotImplementedError("Use aget_tuple")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Synchronous checkpoint list — not supported; always raises.

        Raises:
            NotImplementedError: Always. Use :meth:`alist` instead.
        """
        raise NotImplementedError("Use alist")

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return the latest (or specific) checkpoint for a thread."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        async with self._pool.acquire() as conn:
            if checkpoint_id:
                row = await conn.fetchrow(
                    "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata "
                    "FROM langgraph_checkpoints "
                    "WHERE thread_id=$1 AND checkpoint_ns=$2 AND checkpoint_id=$3",
                    thread_id, checkpoint_ns, checkpoint_id,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata "
                    "FROM langgraph_checkpoints "
                    "WHERE thread_id=$1 AND checkpoint_ns=$2 "
                    "ORDER BY checkpoint_id DESC LIMIT 1",
                    thread_id, checkpoint_ns,
                )

            if not row:
                return None

            chk_id: str = row["checkpoint_id"]
            chk: Checkpoint = self.serde.loads_typed(
                (row["type"], bytes(row["checkpoint"]))
            )
            metadata: CheckpointMetadata = self.serde.loads_typed(
                (row["metadata_type"], bytes(row["metadata"]))
            )

            # Reload full channel_values from blob table
            chk["channel_values"] = await self._load_blobs(
                conn, thread_id, checkpoint_ns, chk.get("channel_versions", {})
            )

            # Pending writes
            write_rows = await conn.fetch(
                "SELECT task_id, channel, type, blob FROM langgraph_checkpoint_writes "
                "WHERE thread_id=$1 AND checkpoint_ns=$2 AND checkpoint_id=$3 "
                "ORDER BY idx",
                thread_id, checkpoint_ns, chk_id,
            )
            pending_writes = [
                (
                    r["task_id"],
                    r["channel"],
                    self.serde.loads_typed((r["type"], bytes(r["blob"]))),
                )
                for r in write_rows
            ]

            parent_id = row["parent_checkpoint_id"]
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": chk_id,
                    }
                },
                checkpoint=chk,
                metadata=metadata,
                pending_writes=pending_writes,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_id,
                        }
                    }
                    if parent_id
                    else None
                ),
            )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Yield checkpoints for a thread, newest first."""
        if not config:
            return
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")

        sql = (
            "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata "
            "FROM langgraph_checkpoints "
            "WHERE thread_id=$1 AND checkpoint_ns=$2 "
            "ORDER BY checkpoint_id DESC"
        )
        params: list = [thread_id, checkpoint_ns]
        if limit:
            sql += f" LIMIT {int(limit)}"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            for row in rows:
                chk: Checkpoint = self.serde.loads_typed(
                    (row["type"], bytes(row["checkpoint"]))
                )
                chk["channel_values"] = await self._load_blobs(
                    conn, thread_id, checkpoint_ns, chk.get("channel_versions", {})
                )
                metadata: CheckpointMetadata = self.serde.loads_typed(
                    (row["metadata_type"], bytes(row["metadata"]))
                )
                parent_id = row["parent_checkpoint_id"]
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row["checkpoint_id"],
                        }
                    },
                    checkpoint=chk,
                    metadata=metadata,
                    pending_writes=[],
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_id,
                            }
                        }
                        if parent_id
                        else None
                    ),
                )

    # ── Write ─────────────────────────────────────────────────────────────────

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        raise NotImplementedError("Use aput")

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        """Persist a checkpoint after a node completes."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        parent_id = config["configurable"].get("checkpoint_id")

        c = checkpoint.copy()
        channel_values: dict = c.pop("channel_values", {})

        chk_type, chk_bytes = self.serde.dumps_typed(c)
        full_metadata = get_checkpoint_metadata(config, metadata)
        meta_type, meta_bytes = self.serde.dumps_typed(full_metadata)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Persist each new channel blob
                for channel, version in (new_versions or {}).items():
                    if channel in channel_values:
                        val_type, val_bytes = self.serde.dumps_typed(channel_values[channel])
                    else:
                        val_type, val_bytes = "empty", b""
                    await conn.execute(
                        """
                        INSERT INTO langgraph_checkpoint_blobs
                            (thread_id, checkpoint_ns, channel, version, type, blob)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
                        """,
                        thread_id, checkpoint_ns, channel, str(version),
                        val_type, val_bytes,
                    )

                # Persist checkpoint row
                await conn.execute(
                    """
                    INSERT INTO langgraph_checkpoints
                        (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                         type, checkpoint, metadata_type, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                        DO UPDATE SET
                            checkpoint     = EXCLUDED.checkpoint,
                            metadata_type  = EXCLUDED.metadata_type,
                            metadata       = EXCLUDED.metadata
                    """,
                    thread_id, checkpoint_ns, checkpoint["id"], parent_id,
                    chk_type, chk_bytes, meta_type, meta_bytes,
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError("Use aput_writes")

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending writes (intermediate results within a node)."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for idx, (channel, value) in enumerate(writes):
                    val_type, val_bytes = self.serde.dumps_typed(value)
                    await conn.execute(
                        """
                        INSERT INTO langgraph_checkpoint_writes
                            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                             channel, type, blob, task_path)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                            DO UPDATE SET type = EXCLUDED.type, blob = EXCLUDED.blob
                        """,
                        thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                        channel, val_type, val_bytes, task_path,
                    )

    # ── Versioning ────────────────────────────────────────────────────────────

    def get_next_version(self, current: str | None, channel: Any) -> str:
        """Monotonically increasing version string — matches MemorySaver behaviour."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        next_h = random.random()  # noqa: S311 — not security-sensitive
        return f"{next_v:032}.{next_h:016}"

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def delete_thread(self, thread_id: str) -> None:
        """Remove all checkpoint data for a thread (e.g. after artifact deletion)."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM langgraph_checkpoint_writes WHERE thread_id=$1", thread_id
                )
                await conn.execute(
                    "DELETE FROM langgraph_checkpoint_blobs WHERE thread_id=$1", thread_id
                )
                await conn.execute(
                    "DELETE FROM langgraph_checkpoints WHERE thread_id=$1", thread_id
                )

    async def adelete_thread(self, thread_id: str) -> None:
        await self.delete_thread(thread_id)

    async def has_checkpoint(self, thread_id: str) -> bool:
        """Return True if any checkpoint exists for this thread_id (artifact)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM langgraph_checkpoints WHERE thread_id=$1 LIMIT 1",
                thread_id,
            )
            return row is not None


# ── Module-level singleton ────────────────────────────────────────────────────
# Lazily initialised on first call to get_checkpointer().
# Double-checked locking prevents duplicate pool creation under concurrent startup.

_checkpointer: AsyncPostgresCheckpointer | None = None
_checkpointer_lock: asyncio.Lock | None = None


def _get_checkpointer_lock() -> asyncio.Lock:
    """Return (creating on first call) the module-level init lock.

    Defined as a function rather than a module-level variable so the Lock is
    always created in the running event loop — required for Python 3.10+.
    """
    global _checkpointer_lock
    if _checkpointer_lock is None:
        _checkpointer_lock = asyncio.Lock()
    return _checkpointer_lock


async def get_checkpointer() -> AsyncPostgresCheckpointer:
    """Return the module-level checkpointer, initialising it on first call.

    Uses the application database URL (stripped of the SQLAlchemy driver prefix)
    so no extra configuration is required. Thread-safe via double-checked locking
    so concurrent startup paths cannot create duplicate connection pools.

    Returns:
        The singleton :class:`AsyncPostgresCheckpointer` instance.
    """
    global _checkpointer
    if _checkpointer is not None:  # fast path — no lock needed
        return _checkpointer

    async with _get_checkpointer_lock():
        if _checkpointer is not None:  # re-check under lock
            return _checkpointer

        from app.core.config import settings

        dsn = settings.database_url
        # Strip the SQLAlchemy driver prefix so asyncpg can parse it
        for prefix in ("postgresql+asyncpg://", "postgres+asyncpg://"):
            if dsn.startswith(prefix):
                dsn = "postgresql://" + dsn[len(prefix):]
                break

        _checkpointer = await AsyncPostgresCheckpointer.create(dsn)
        log.info("checkpointer: PostgreSQL checkpoint store initialised")
        return _checkpointer
