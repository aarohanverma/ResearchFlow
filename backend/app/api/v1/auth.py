"""Auth router — register, login, me.

Defensive choices documented here so they survive future edits:

* **Email normalisation** — emails are lowercased + stripped at the
  schema layer. The DB stores the lowercased form.
* **Account-enumeration defence** — login returns the same 401 message
  for both "no such user" and "wrong password", and runs a dummy bcrypt
  verify on the missing-user branch so response timing is comparable.
* **Password length** — bcrypt's 72-byte input limit is enforced
  centrally in ``app.core.security`` and surfaces as HTTP 400 here.
"""

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.deps import CurrentUserID, DBSession
from app.core.security import (
    PasswordTooLongError,
    constant_time_dummy_verify,
    create_access_token,
    hash_password,
    verify_password,
)
from app.repositories.user import UserRepository
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(user_id: str) -> TokenResponse:
    """Mint a token + expiry pair so clients can plan refresh.

    The ``expires_in`` field matches the OAuth2 convention; keeping it
    in the response means migrating to a managed identity provider
    later is a one-line client change.
    """
    return TokenResponse(
        access_token=create_access_token(user_id),
        expires_in=settings.jwt_expire_minutes * 60,
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: DBSession):
    """Register a new user account and return a JWT access token.

    Raises:
        HTTPException: 400 if the password exceeds bcrypt's 72-byte input
            limit; 409 if the email is already registered.
    """
    repo = UserRepository(db)
    if await repo.get_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    try:
        hashed = hash_password(body.password)
    except PasswordTooLongError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    user = await repo.create(body.email, hashed, body.display_name)
    await db.commit()
    return _token_response(str(user.id))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DBSession):
    """Authenticate a user and return a JWT access token.

    Uses a uniform error message and a dummy bcrypt verify on the
    user-not-found branch so timing differences do not leak whether an
    email is registered.
    """
    repo = UserRepository(db)
    user = await repo.get_by_email(body.email)
    if user is None:
        constant_time_dummy_verify()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return _token_response(str(user.id))


@router.get("/me", response_model=UserResponse)
async def me(user_id: CurrentUserID, db: DBSession):
    """Return the profile of the currently authenticated user.

    Args:
        user_id: UUID of the authenticated user, injected by the auth dependency.
        db: Injected async database session.

    Returns:
        A ``UserResponse`` with the user's public profile fields.

    Raises:
        HTTPException: 404 if no user row exists for the given ID.
    """
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)
