"""JWT creation/verification and password hashing utilities.

Design notes
------------

* **bcrypt 72-byte limit** — bcrypt silently truncates any input longer
  than 72 bytes, which makes long passwords weaker than they look. We
  reject oversized inputs explicitly so the user receives a clear error
  and we never store a hash derived from a truncated input.
* **Issuer claim** — every token carries ``iss="researchflow"``, and the
  decoder validates it. Single-issuer today; the field is in place so a
  future deployment behind a gateway can reject tokens minted by an
  unrelated service that happens to share the secret.
* **Required claims** — ``decode_access_token`` requires ``sub``,
  ``exp``, ``iat`` to be present so a malformed token cannot bypass
  expiry or identity checks.
* **Constant-time dummy verify** — login can call
  ``constant_time_dummy_verify`` on the user-not-found branch to keep
  timing uniform and block account-enumeration attacks.
* **Cloud parity** — all knobs (secret, algorithm, expiry, issuer) come
  from ``Settings``, so moving from local to a managed deployment is
  just an environment-variable change.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt as _bcrypt
from jose import JWTError, jwt  # noqa: F401 — re-exported for callers

from app.core.config import settings

# bcrypt operates on the first 72 bytes of input. Anything longer would
# be silently truncated — reject up front instead so callers see the
# limit explicitly. Note: a single emoji is 4 UTF-8 bytes, so this cap
# is generous in practice.
_BCRYPT_MAX_BYTES = 72

# Issuer claim. Single-issuer today, but stamping it now means a future
# multi-service deployment can validate the source without a migration.
JWT_ISSUER = "researchflow"


class PasswordTooLongError(ValueError):
    """Raised when a password exceeds bcrypt's 72-byte input limit."""


def _password_bytes(plain: str) -> bytes:
    """Encode the password to bytes and reject oversized inputs.

    Centralising this guard means ``hash_password`` and ``verify_password``
    apply the same rule, so a password that registered cleanly always
    verifies cleanly.
    """
    encoded = (plain or "").encode("utf-8")
    if len(encoded) > _BCRYPT_MAX_BYTES:
        raise PasswordTooLongError(
            f"Password exceeds {_BCRYPT_MAX_BYTES}-byte limit "
            f"(got {len(encoded)} bytes after UTF-8 encoding)."
        )
    return encoded


def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt.

    Raises:
        PasswordTooLongError: when the UTF-8 encoded password exceeds
            bcrypt's 72-byte limit. Callers should translate this to a
            400-level HTTP response.
    """
    return _bcrypt.hashpw(_password_bytes(plain), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plain-text password against a stored bcrypt hash.

    Returns ``False`` (rather than raising) for malformed inputs so the
    login path can stay branch-free and timing-uniform.
    """
    try:
        pw_bytes = _password_bytes(plain)
    except PasswordTooLongError:
        return False
    try:
        return _bcrypt.checkpw(pw_bytes, (hashed or "").encode("utf-8"))
    except (ValueError, TypeError):
        # Corrupted or empty hash — treat as a failed verification.
        return False


# Pre-computed dummy hash used by ``constant_time_dummy_verify`` so the
# login handler can perform a bcrypt round even when the user does not
# exist. Computed once at import — bcrypt salt generation here is slow
# but happens only at module load.
_DUMMY_HASH = _bcrypt.hashpw(b"x" * 32, _bcrypt.gensalt()).decode()


def constant_time_dummy_verify() -> None:
    """Run a bcrypt verify against a throwaway hash.

    Call on the failure branch of login when the user does not exist so
    that successful and failed lookups take comparable time. Defends
    against account-enumeration via response-timing. The return value
    is intentionally discarded; any exception is swallowed so timing
    defence never affects correctness.
    """
    try:
        _bcrypt.checkpw(b"x" * 32, _DUMMY_HASH.encode("utf-8"))
    except Exception:
        pass


def create_access_token(subject: str | Any, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token for the given subject.

    The token carries ``sub`` (user id), ``iss``, ``iat``, ``exp``.

    Args:
        subject: The token subject — typically a user UUID string.
        expires_delta: Optional custom expiry duration. Defaults to
            ``settings.jwt_expire_minutes``.
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.jwt_expire_minutes))
    payload = {
        "sub": str(subject),
        "iss": JWT_ISSUER,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT.

    Validates issuer + requires ``sub``/``exp``/``iat`` claims. Raises
    ``JWTError`` on signature mismatch, expiry, wrong issuer, or
    missing required claims.
    """
    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        issuer=JWT_ISSUER,
        options={"require": ["sub", "exp", "iat"]},
    )
