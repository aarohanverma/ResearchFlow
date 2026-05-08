"""FastAPI dependency injection — DB sessions, current user, adapters."""

from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.core.tracking import current_user_id as _current_user_id_ctx
from app.db.session import async_session_factory

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_db() -> AsyncSession:
    """Yields a per-request async DB session — always closed after the request."""
    async with async_session_factory() as session:
        yield session


DBSession = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> UUID:
    """Decodes JWT and returns the user UUID.  Never leaks internal errors."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise credentials_exc
        uid = UUID(user_id_str)
        # Stash on the request-local contextvar so token-usage tracking can
        # attribute every LLM call made during this request to this user.
        _current_user_id_ctx.set(uid)
        return uid
    except (JWTError, ValueError):
        raise credentials_exc


CurrentUserID = Annotated[UUID, Depends(get_current_user_id)]
