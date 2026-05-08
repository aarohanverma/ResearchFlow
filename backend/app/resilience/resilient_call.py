"""Resilience layer — four-tier safety net for every external call.

Tier 1: Retry with exponential backoff + jitter (tenacity)
Tier 2: Provider fallback chain (LLMAdapter handles this internally)
Tier 3: Circuit breaker (pybreaker) — opens on N consecutive failures
Tier 4: Graceful degradation — caller gets a DegradedResult instead of an exception
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
    """Returned instead of raising when all tiers are exhausted."""
    error: str
    degraded: bool = True
    data: Any = None


def _is_retryable(exc: Exception) -> bool:
    """Only retry on transient errors — never on 4xx auth / bad-request."""
    import httpx
    import openai
    import anthropic

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return True
    return False


# One circuit breaker per provider name — shared across process lifetime.
_breakers: dict[str, pybreaker.CircuitBreaker] = {}


def _get_breaker(provider: str) -> pybreaker.CircuitBreaker:
    """Return (creating if necessary) the circuit breaker for a given provider name."""
    if provider not in _breakers:
        _breakers[provider] = pybreaker.CircuitBreaker(
            fail_max=5,          # open after 5 consecutive failures
            reset_timeout=60,    # half-open after 60 s
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
    """Execute fn with retry, circuit-breaker, fallback, and graceful degradation.

    Args:
        provider: Provider name for circuit-breaker keying and logging.
        fn: The async callable to protect.
        max_attempts: Max retry attempts (default 3).
        fallback_fn: Async callable used after retries are exhausted.
        degrade_value: Returned if fallback_fn also fails (graceful degradation).
        workflow: LangSmith tracing context label.
        node: LangSmith node label.
    """
    breaker = _get_breaker(provider)
    start = time.monotonic()

    async def _attempt():
        """Run ``fn`` with exponential-jitter retries inside the circuit breaker."""
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
        ):
            with attempt:
                return await fn(*args, **kwargs)

    try:
        # Tier 1 + 3: retry inside circuit breaker
        try:
            result = breaker.call(lambda: asyncio.get_event_loop().run_until_complete(_attempt()))
        except TypeError:
            # pybreaker doesn't support async natively — wrap coroutine
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
    """Decorator version of resilient_call."""
    def decorator(fn: Callable) -> Callable:
        """Wrap a callable with resilient retry/fallback logic."""
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            """Execute the wrapped callable through resilient_call."""
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
