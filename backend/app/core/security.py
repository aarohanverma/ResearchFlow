"""JWT creation/verification and password hashing utilities."""

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt as _bcrypt
from jose import JWTError, jwt

from app.core.config import settings


def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt.

    Args:
        plain: The plain-text password string to hash.

    Returns:
        A bcrypt-hashed password string suitable for database storage.
    """
    return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plain-text password against a stored bcrypt hash.

    Args:
        plain: The plain-text password to verify.
        hashed: The bcrypt hash string previously returned by ``hash_password``.

    Returns:
        ``True`` if the password matches the hash, ``False`` otherwise.
    """
    return _bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(subject: str | Any, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token for the given subject.

    Args:
        subject: The token subject — typically a user UUID string.
        expires_delta: Optional custom expiry duration. Defaults to
            ``settings.jwt_expire_minutes`` (7 days).

    Returns:
        A signed JWT string encoded with ``settings.jwt_algorithm``.
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    payload = {"sub": str(subject), "exp": expire, "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """Raises JWTError on invalid or expired token."""
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
