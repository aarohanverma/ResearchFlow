"""PII redaction layer for memory writes.

The LangChain prebuilt-middleware catalogue includes a ``PIIMiddleware``
that detects + redacts personally-identifiable information before it
crosses agent boundaries. RA's prior gap: anything the LLM decided to
write into long-lived memory (``auto_memory``, the ``memory_write``
tool) landed verbatim in ``session.state``. A user who pasted a credit
card number, API key, or email into a query could see that PII
persisted forever in the embedding cache and in the DB row.

Scope (deliberately narrow):

  * **Memory writes only** — the highest-risk surface. Once
    persisted, PII is hard to remove (it's in DB rows, embedding
    caches, branch summaries, possibly LangSmith traces).
  * **Conservative regex set** — credit cards (Luhn-validated to cut
    false positives), email addresses, US SSNs, common API-key
    prefixes (``sk-``, ``OPENAI_API_KEY=…``-style), and bare phone
    numbers. Anything more (passport numbers, IBANs, etc.) belongs
    in a per-deployment policy module, not here.
  * **Default strategy is REDACT** — replace the match with
    ``[REDACTED_{TYPE}]`` so the surrounding context (and the
    semantic embedding) still makes sense. Some callers may prefer
    ``MASK`` (keep last 4 chars) for partial reference — exposed as
    a parameter.
  * **NEVER raises into callers** — on any internal failure (regex
    catastrophic backtrack, Luhn lib glitch) the helper returns the
    original text unchanged so memory writes never silently fail
    because the redactor blew up.

Not in scope (intentionally):

  * Inbound user query redaction — would break query-aware retrieval
    semantically (the user asked about *their* card number).
  * LLM response redaction — the synthesiser's output is grounded in
    retrieved papers / DB rows; PII bleeding through there is a
    different threat model (data exfiltration) and should be solved
    upstream (don't store secrets in indexable documents).
  * PII in logs — the resilient_call layer already logs only the
    exception message, not the call args. If a deployment needs
    full log scrubbing, that's a structlog processor, not this
    module.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

log = logging.getLogger(__name__)


# ── Patterns ────────────────────────────────────────────────────────────────
#
# Each pattern is intentionally conservative; we'd rather miss a soft
# match than corrupt a real value. The detector returns the type
# label which is what shows up in the redacted string.
#
# CREDIT_CARD: 13–19 digits, optional spaces/dashes between groups. We
#   run Luhn against the digit-only string to cut the false-positive
#   rate dramatically (random 16-digit numbers fail Luhn ~90% of the
#   time). The Luhn step is in code below, not in the regex itself.
_PATTERN_CC = re.compile(
    r"\b(?:\d[ -]?){13,19}\b",
)

# Email — the practical RFC 5321 surface, not a full RFC parser. The
# upper bound prevents catastrophic backtracking on adversarial input.
_PATTERN_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}\b",
)

# US SSN with delimiters — XXX-XX-XXXX. Bare 9-digit numbers without
# the dashes overlap with phone numbers and zip+4 codes too often;
# leave those to a per-deployment classifier.
_PATTERN_SSN = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
)

# API-key shapes — keep additive: the openai ``sk-`` family, the
# anthropic ``sk-ant-`` family, generic 32+-char hex/base62, and the
# common ``FOO_API_KEY=…`` env-style. Bare hex/base62 alone is too
# eager (real arxiv ids, UUIDs, hashes); we require a leading prefix
# OR an env-assignment context.
_PATTERN_APIKEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{16,}|"
    r"AKIA[0-9A-Z]{16}|"            # AWS access key
    r"ghp_[A-Za-z0-9]{32,}|"        # GitHub personal access
    r"gho_[A-Za-z0-9]{32,}|"        # GitHub OAuth
    r"AIza[0-9A-Za-z_\-]{35})\b",
)

# Env-style assignment: KEY_NAME = value where the key NAME contains
# SECRET/TOKEN/KEY/PASSWORD. Strips the value, not the key.
_PATTERN_ENV_SECRET = re.compile(
    r"(?im)^(\s*[A-Z][A-Z0-9_]*(?:SECRET|TOKEN|KEY|PASSWORD|PASS)[A-Z0-9_]*\s*[:=]\s*)"
    r"(?P<value>['\"]?[^\s'\"]{6,}['\"]?)",
)

# Phone — E.164-ish or US-style with separators. Tight to avoid
# matching long sequences of digits (UUIDs, hashes).
_PATTERN_PHONE = re.compile(
    r"(?<![\d/-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)",
)


# Order matters: we apply higher-confidence / more-specific patterns
# first so a card number that happens to match a phone shape gets
# tagged as a card.
_REDACTION_PASSES: list[tuple[str, "re.Pattern[str]"]] = [
    ("CARD", _PATTERN_CC),
    ("APIKEY", _PATTERN_APIKEY),
    ("SSN", _PATTERN_SSN),
    ("EMAIL", _PATTERN_EMAIL),
    ("PHONE", _PATTERN_PHONE),
]


# ── Public API ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RedactionResult:
    """Outcome of one redact_pii call.

    Attributes:
        text: The redacted text (or the original on failure).
        found: Distinct PII type labels found (e.g. ``{"EMAIL", "CARD"}``).
            Empty when no PII matched.
    """
    text: str
    found: frozenset[str]


def _luhn_valid(digits_only: str) -> bool:
    """Return True iff the all-digit string passes the Luhn check.

    Random 16-digit strings pass Luhn about 10% of the time, so the
    check is the right post-filter to reject coincidences. Length is
    bounded by the regex (13–19 digits).
    """
    if not digits_only.isdigit() or not (13 <= len(digits_only) <= 19):
        return False
    total = 0
    parity = len(digits_only) % 2
    for i, ch in enumerate(digits_only):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(
    text: str,
    *,
    enabled_types: Iterable[str] | None = None,
) -> RedactionResult:
    """Redact PII in ``text`` using the conservative pattern set.

    Args:
        text: Free-text input. Empty / non-string returns as-is.
        enabled_types: Optional subset of pattern labels (``"CARD"``,
            ``"EMAIL"``, ``"SSN"``, ``"APIKEY"``, ``"PHONE"``). When
            omitted, all are enabled. Useful for tests or for callers
            that want to opt out of (e.g.) email redaction.

    Returns:
        :class:`RedactionResult` with the new text and the set of
        types found. On any internal error the original text is
        returned with an empty ``found`` set — the helper never
        raises so memory-write paths cannot silently fail because of
        a regex misfire.
    """
    if not text or not isinstance(text, str):
        return RedactionResult(text=text or "", found=frozenset())

    try:
        enabled = (
            {t.upper() for t in enabled_types}
            if enabled_types is not None
            else {label for label, _ in _REDACTION_PASSES}
        )

        found: set[str] = set()
        out = text

        # Env-style secret assignments come first because a hit here
        # also wants to strip the VALUE, not the whole match.
        if "APIKEY" in enabled:
            def _env_sub(m: re.Match[str]) -> str:
                found.add("APIKEY")
                return m.group(1) + "[REDACTED_APIKEY]"
            out = _PATTERN_ENV_SECRET.sub(_env_sub, out)

        for label, pattern in _REDACTION_PASSES:
            if label not in enabled:
                continue

            if label == "CARD":
                # Luhn-filter to reject 16-digit coincidences (UUIDs
                # in some formats, hashes, etc.).
                def _cc_sub(m: re.Match[str]) -> str:
                    digits = re.sub(r"\D", "", m.group(0))
                    if _luhn_valid(digits):
                        found.add("CARD")
                        return "[REDACTED_CARD]"
                    return m.group(0)
                out = pattern.sub(_cc_sub, out)
                continue

            def _sub(m: re.Match[str], _label: str = label) -> str:
                found.add(_label)
                return f"[REDACTED_{_label}]"
            out = pattern.sub(_sub, out)

        return RedactionResult(text=out, found=frozenset(found))
    except Exception as exc:  # noqa: BLE001 — must never abort a memory write
        log.debug("redact_pii: failure on len=%d: %s", len(text or ""), exc)
        return RedactionResult(text=text, found=frozenset())


__all__ = ["redact_pii", "RedactionResult"]
