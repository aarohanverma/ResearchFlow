"""Resilience layer — four-tier safety net for every external call.

Tier 1: Retry with exponential backoff + jitter (tenacity)
Tier 2: Provider fallback chain (supplied via ``fallback_fn``)
Tier 3: Circuit breaker (pybreaker) — opens on N consecutive failures
Tier 4: Graceful degradation — caller gets ``degrade_value`` instead of an exception

Circuit-breaker notes
---------------------
pybreaker's ``call()`` method is synchronous-only.  We use ``call_async()``
(available in pybreaker ≥ 1.0) with an ``AttributeError`` guard for older
installs.  The guard avoids the previous ``run_until_complete()`` anti-pattern
which raised ``RuntimeError: This event loop is already running`` on every call
from an async context, effectively bypassing the circuit breaker entirely.

The ``_breakers`` dict is guarded by a module-level ``asyncio.Lock`` so
concurrent first-calls cannot produce duplicate breaker objects.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import pybreaker
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)

F = TypeVar("F")


@dataclass
class DegradedResult:
    """Returned instead of raising when all tiers are exhausted.

    Attributes:
        error: Human-readable description of the failure.
        degraded: Always ``True`` — lets callers distinguish a real result
            from a degraded placeholder.
        data: Optional partial data carried from earlier tiers.
    """

    error: str
    degraded: bool = True
    data: Any = None


def _is_retryable(exc: Exception) -> bool:
    """Return ``True`` for transient errors that warrant a retry.

    Never retries on 4xx authentication / bad-request errors.

    Args:
        exc: The exception raised by the protected callable.

    Returns:
        ``True`` if the error is transient (network, rate limit, 5xx server).
    """
    try:
        import httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
            return True
    except ImportError:
        pass
    try:
        import openai
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
            return True
    except ImportError:
        pass
    try:
        import anthropic
        if isinstance(exc, anthropic.RateLimitError):
            return True
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
            return True
    except ImportError:
        pass
    return False


# ── Per-provider circuit breakers ─────────────────────────────────────────────
# One breaker per provider name, shared across the process lifetime.
# Protected by a module-level lock to prevent duplicate creation under
# concurrent first-calls.
#
# The lock is initialised eagerly at import time (not lazily) to avoid a
# race where two coroutines simultaneously see `_breakers_lock is None`,
# both create a new asyncio.Lock(), and one is immediately orphaned.
# asyncio.Lock() is cheap — creating it at module load is safe.

_breakers: dict[str, pybreaker.CircuitBreaker] = {}
_breakers_lock = asyncio.Lock()


async def _get_breaker_async(provider: str) -> pybreaker.CircuitBreaker:
    """Return the circuit breaker for ``provider``, creating it if needed.

    Args:
        provider: Short provider name (e.g. ``"openai"``).

    Returns:
        The :class:`pybreaker.CircuitBreaker` instance for that provider.
    """
    if provider in _breakers:  # fast path — no lock needed for reads
        return _breakers[provider]
    async with _breakers_lock:
        if provider not in _breakers:  # re-check under lock
            _breakers[provider] = pybreaker.CircuitBreaker(
                fail_max=5,       # open after 5 consecutive failures
                reset_timeout=60, # half-open after 60 s
                name=provider,
            )
    return _breakers[provider]


async def resilient_call(
    provider: str,
    fn: Callable,
    *args,
    max_attempts: int = 3,
    fallback_fn: Callable | None = None,
    degrade_value: Any = None,
    workflow: str = "",
    node: str = "",
    **kwargs,
) -> Any:
    """Execute ``fn`` with retry, circuit-breaker, fallback, and graceful degradation.

    Args:
        provider: Provider name used for circuit-breaker keying and log context.
        fn: Async callable to protect.
        max_attempts: Maximum retry attempts (default 3).
        fallback_fn: Async callable invoked when retries are exhausted.
        degrade_value: Returned when both ``fn`` and ``fallback_fn`` fail.
        workflow: Workflow name for structured logging.
        node: Node name for structured logging.
        *args: Positional arguments forwarded to ``fn`` (and ``fallback_fn``).
        **kwargs: Keyword arguments forwarded to ``fn`` (and ``fallback_fn``).

    Returns:
        Result of ``fn``, ``fallback_fn``, or ``degrade_value`` (in that order).

    Raises:
        The last exception raised by ``fn`` when no fallback or degrade value
        is configured and all tiers are exhausted.
    """
    breaker = await _get_breaker_async(provider)
    start = time.monotonic()

    async def _attempt():
        """Run ``fn`` with exponential-jitter retries."""
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
        ):
            with attempt:
                return await fn(*args, **kwargs)

    try:
        # Tier 1 + 3: retry inside the circuit breaker.
        # Use call_async (pybreaker ≥ 1.0) with a fallback for older installs.
        # The previous breaker.call(lambda: run_until_complete(...)) pattern
        # always raised RuntimeError inside an async context and was inoperative.
        try:
            result = await breaker.call_async(_attempt)
        except AttributeError:
            # Older pybreaker without call_async — bypass breaker, keep retries
            result = await _attempt()

        log.debug(
            "resilient_call ok provider=%s workflow=%s node=%s latency_ms=%d",
            provider, workflow, node, int((time.monotonic() - start) * 1000),
        )
        return result

    except (RetryError, Exception) as primary_exc:
        log.warning(
            "resilient_call exhausted provider=%s workflow=%s node=%s err=%s",
            provider, workflow, node, primary_exc,
        )

        # Tier 2: try fallback provider
        if fallback_fn is not None:
            try:
                log.info("resilient_call falling back workflow=%s node=%s", workflow, node)
                return await fallback_fn(*args, **kwargs)
            except Exception as fallback_exc:
                log.error(
                    "resilient_call fallback also failed workflow=%s node=%s err=%s",
                    workflow, node, fallback_exc,
                )

        # Tier 4: graceful degradation
        if degrade_value is not None:
            log.warning("resilient_call degrading workflow=%s node=%s", workflow, node)
            return degrade_value

        raise  # re-raise primary so callers can record error_metadata


def with_resilience(
    provider: str,
    max_attempts: int = 3,
    fallback_fn: Callable | None = None,
    degrade_value: Any = None,
    workflow: str = "",
    node: str = "",
):
    """Decorator version of :func:`resilient_call`.

    Args:
        provider: Provider name for circuit-breaker keying.
        max_attempts: Maximum retry attempts.
        fallback_fn: Async fallback callable.
        degrade_value: Value returned when all tiers are exhausted.
        workflow: Workflow label for structured logging.
        node: Node label for structured logging.

    Returns:
        Decorator that wraps an async callable with the four-tier safety net.
    """
    def decorator(fn: Callable) -> Callable:
        """Wrap a callable with resilient retry/fallback logic."""
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await resilient_call(
                provider=provider,
                fn=fn,
                *args,
                max_attempts=max_attempts,
                fallback_fn=fallback_fn,
                degrade_value=degrade_value,
                workflow=workflow,
                node=node,
                **kwargs,
            )
        return wrapper
    return decorator
