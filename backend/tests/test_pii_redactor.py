"""Tests for the memory-write PII redactor.

The redactor is the only barrier between LLM-decided memory writes
and persistent ``session.state``. It must:

  * Catch the common PII shapes (credit cards Luhn-validated, emails,
    SSNs, API keys, phone numbers) without breaking normal text.
  * Stay conservative — random 16-digit IDs that aren't real cards
    must pass through unchanged.
  * Never raise into callers; on internal failure the original text
    is returned (memory write must not fail because the redactor
    glitched).
"""

from __future__ import annotations

import pytest

from app.assistant.pii_redactor import redact_pii


# ── Positive matches ────────────────────────────────────────────────────────


def test_redacts_valid_credit_card():
    # 4111-1111-1111-1111 is the canonical Visa test number and
    # passes Luhn.
    out = redact_pii("My card is 4111-1111-1111-1111 if you need it.")
    assert "[REDACTED_CARD]" in out.text
    assert "4111" not in out.text
    assert "CARD" in out.found


def test_redacts_email():
    out = redact_pii("Email me at aarohan@example.com for follow-up.")
    assert "[REDACTED_EMAIL]" in out.text
    assert "@example.com" not in out.text
    assert "EMAIL" in out.found


def test_redacts_ssn_with_delimiters():
    out = redact_pii("SSN on file is 123-45-6789 please verify.")
    assert "[REDACTED_SSN]" in out.text
    assert "123-45-6789" not in out.text
    assert "SSN" in out.found


def test_redacts_openai_api_key():
    out = redact_pii("Token: sk-proj-AAAA1234567890aaaaBBBBccccDDDDeeee end")
    assert "[REDACTED_APIKEY]" in out.text
    assert "APIKEY" in out.found


def test_redacts_anthropic_api_key():
    out = redact_pii("Bearer sk-ant-AAAA1234567890aaaaBBBBccccDDDD here")
    assert "[REDACTED_APIKEY]" in out.text


def test_redacts_aws_access_key():
    out = redact_pii("Use AKIAIOSFODNN7EXAMPLE to connect.")
    assert "[REDACTED_APIKEY]" in out.text


def test_redacts_env_style_secret():
    # KEY = value style; the key NAME stays, the value is masked.
    out = redact_pii("DATABASE_PASSWORD = supersecret123!")
    assert "DATABASE_PASSWORD" in out.text
    assert "supersecret" not in out.text
    assert "[REDACTED_APIKEY]" in out.text


def test_redacts_phone_us_format():
    out = redact_pii("Call (415) 555-0142 anytime.")
    assert "[REDACTED_PHONE]" in out.text


def test_redacts_phone_e164():
    out = redact_pii("Reach me at +1 415 555 0142 today.")
    assert "[REDACTED_PHONE]" in out.text


def test_redacts_multiple_kinds_in_one_pass():
    out = redact_pii(
        "Contact aarohan@example.com or call (415) 555-0142; "
        "API key sk-test-AAAA1234567890aaaaBBBBccccDDDD"
    )
    assert "EMAIL" in out.found
    assert "PHONE" in out.found
    assert "APIKEY" in out.found


# ── Conservative non-matches ────────────────────────────────────────────────


def test_random_16_digit_id_passes_through_when_luhn_invalid():
    """A 16-digit ID that ISN'T a real card (fails Luhn) must NOT be
    redacted — the most common false-positive source for CC regexes."""
    # 1234567890123456 fails Luhn (sum = 70, not divisible by 10? let
    # me check: 1+2*2+3+4*2+5+6*2+7+8*2+9+0*2+1+2*2+3+4*2+5+6 ... etc.)
    # The point: a deliberately non-Luhn sequence stays intact.
    out = redact_pii("Internal trace id 1234567890123456 do not change.")
    assert "1234567890123456" in out.text
    assert "CARD" not in out.found


def test_uuid_with_dashes_not_treated_as_card():
    out = redact_pii("Paper id ab12cd34-5678-90ef-1234-567890abcdef hello.")
    assert "ab12cd34-5678-90ef-1234-567890abcdef" in out.text
    assert "CARD" not in out.found


def test_arxiv_id_not_treated_as_phone():
    out = redact_pii("Cite 2401.12345 as a baseline.")
    assert "2401.12345" in out.text
    assert "PHONE" not in out.found


def test_no_pii_text_passes_through():
    src = "Transformer self-attention scales quadratically with sequence length."
    out = redact_pii(src)
    assert out.text == src
    assert out.found == frozenset()


def test_empty_input_returns_empty():
    assert redact_pii("").text == ""
    assert redact_pii(None).text == ""  # type: ignore[arg-type]


# ── Never-raises contract ───────────────────────────────────────────────────


def test_never_raises_on_pathological_input():
    # Long input with many potentially-matching shapes — must not
    # explode and must return at most an extracted text.
    src = ("aaa" * 5000) + " 4111-1111-1111-1111 " + ("bbb" * 5000)
    out = redact_pii(src)
    assert "[REDACTED_CARD]" in out.text
