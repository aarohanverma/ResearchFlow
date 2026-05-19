"""Semantic + boundary-aware chunking for RAG indexing.

The default chunking strategy used to be "one chunk per parsed section",
which is semantic-by-section but breaks when a single section is many
thousands of characters — a single embedding then has to summarise the
whole "Results" block, which kills retrieval precision.

This module fixes that with a two-pass approach:

1.  **Section-level semantics** — caller supplies pre-segmented sections
    (abstract, methods, results, …). Section type is preserved on every
    sub-chunk so retrieval can still filter / cite by section.

2.  **Boundary-aware sub-chunking** — when a section exceeds
    ``TARGET_CHARS``, it is split on the *strongest* available
    boundary first (paragraph → sentence → word), falling back only
    when the next-coarser boundary would yield a chunk that is too
    large. We never split mid-sentence unless a single sentence is
    itself larger than ``HARD_CAP_CHARS`` (rare; defensive fallback).

3.  **Overlap** — adjacent sub-chunks share a one-sentence tail so a
    fact that straddles a boundary is still retrievable from either
    side.

The output preserves the existing ``PaperChunk`` row schema, so call-sites
only need to swap the loop body — no migration is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Tunables ──────────────────────────────────────────────────────────────────
# Targets chosen to align with how OpenAI / Gemini / Voyage embedding
# models behave: ~800–1200 chars / ~150–300 tokens is the sweet spot for
# scientific prose. Below ~200 chars the embedding is dominated by
# stopword variance; above ~2000 it averages out too many topics.
TARGET_CHARS = 1100
HARD_CAP_CHARS = 2200
MIN_CHARS = 220       # below this we stop splitting and emit what's left
OVERLAP_SENTENCES = 1  # tail of previous chunk reattached to the next


# Paragraph boundaries: two-or-more consecutive newlines, optionally with
# tabs/spaces in between. Robust to Markdown-style double newlines.
_PARA_RE = re.compile(r"\n[ \t]*\n[ \t\n]*")

# Sentence boundaries: ".", "!", or "?" followed by whitespace and an
# uppercase letter or digit. Avoids splitting on common abbreviations
# (e.g., "Fig.", "et al.", "vs.") by requiring a capital letter after.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class SemanticChunk:
    """One ready-to-embed unit produced by :func:`chunk_section`.

    Maps 1:1 onto a ``PaperChunk`` row — ``section_type`` and
    ``chunk_index`` come from the caller (it owns global indexing across
    sections), ``content`` is the text to embed.
    """

    content: str
    section_type: str


# ── Public API ────────────────────────────────────────────────────────────────


def chunk_section(text: str, section_type: str) -> list[SemanticChunk]:
    """Return a list of boundary-aware sub-chunks for one section.

    Short sections (below ``TARGET_CHARS``) emit a single chunk. Longer
    sections are split paragraph-first, then sentence-first within
    over-large paragraphs. The output always preserves ``section_type``
    so downstream retrieval / citation logic that filters by section
    still works.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= TARGET_CHARS:
        return [SemanticChunk(content=cleaned, section_type=section_type)]

    paragraphs = [p.strip() for p in _PARA_RE.split(cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    # Pass 1: pack paragraphs into target-sized buckets without breaking them.
    buckets: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if len(para) > HARD_CAP_CHARS:
            # Flush whatever we have, then split this paragraph by sentence.
            if current:
                buckets.append("\n\n".join(current))
                current, current_len = [], 0
            buckets.extend(_split_paragraph_by_sentence(para))
            continue
        if current_len + len(para) + 2 > TARGET_CHARS and current:
            buckets.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        buckets.append("\n\n".join(current))

    # Pass 2: add a one-sentence overlap between adjacent buckets so a
    # fact that straddles the boundary still appears verbatim on the
    # next chunk's start (and so its embedding partially shares context).
    overlapped: list[str] = []
    for i, bucket in enumerate(buckets):
        if i == 0 or OVERLAP_SENTENCES <= 0:
            overlapped.append(bucket)
            continue
        prev = buckets[i - 1]
        tail = _last_sentences(prev, OVERLAP_SENTENCES)
        if tail and not bucket.startswith(tail):
            overlapped.append(f"{tail} {bucket}")
        else:
            overlapped.append(bucket)

    # Drop tiny tails — a 50-char chunk pollutes retrieval. Merge with
    # the previous chunk instead.
    merged: list[str] = []
    for chunk in overlapped:
        if merged and len(chunk) < MIN_CHARS:
            merged[-1] = (merged[-1] + "\n\n" + chunk).strip()
        else:
            merged.append(chunk)

    return [SemanticChunk(content=c.strip(), section_type=section_type) for c in merged if c.strip()]


def chunk_sections(sections: list[dict]) -> list[SemanticChunk]:
    """Convenience: run :func:`chunk_section` over a list of sections.

    Each dict must have ``type`` and ``content``. Order is preserved.
    """
    out: list[SemanticChunk] = []
    for sec in sections:
        st = (sec.get("type") or "section").strip() or "section"
        out.extend(chunk_section(sec.get("content") or "", st))
    return out


# ── Internals ─────────────────────────────────────────────────────────────────


def _split_paragraph_by_sentence(paragraph: str) -> list[str]:
    """Split a too-large paragraph along sentence boundaries.

    Greedily packs sentences into ``TARGET_CHARS`` buckets. If a single
    sentence exceeds ``HARD_CAP_CHARS`` (extremely rare in well-formed
    prose; happens with citation-heavy or formula-laden text), it falls
    back to a word-boundary slice as a last resort.
    """
    sentences = _SENT_RE.split(paragraph)
    if not sentences:
        return _hard_split_by_words(paragraph)
    buckets: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > HARD_CAP_CHARS:
            if current:
                buckets.append(" ".join(current))
                current, current_len = [], 0
            buckets.extend(_hard_split_by_words(sent))
            continue
        if current_len + len(sent) + 1 > TARGET_CHARS and current:
            buckets.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        buckets.append(" ".join(current))
    return buckets


def _hard_split_by_words(text: str) -> list[str]:
    """Last-resort word-boundary slicer.

    Used only when a single sentence is larger than ``HARD_CAP_CHARS``
    (e.g., a 3000-char URL-laden footnote). Keeps word boundaries so we
    never split a token mid-character, which would corrupt the embedding.
    """
    words = text.split()
    if not words:
        return []
    buckets: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > TARGET_CHARS and current:
            buckets.append(" ".join(current))
            current, current_len = [], 0
        current.append(word)
        current_len += len(word) + 1
    if current:
        buckets.append(" ".join(current))
    return buckets


def _last_sentences(text: str, n: int) -> str:
    """Return the trailing ``n`` sentences of ``text`` for overlap."""
    if n <= 0:
        return ""
    sentences = _SENT_RE.split(text)
    if not sentences:
        return ""
    tail = sentences[-n:]
    return " ".join(s.strip() for s in tail if s and s.strip())
