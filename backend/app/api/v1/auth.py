"""Auth router — register, login, me."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUserID, DBSession
from app.core.security import create_access_token, hash_password, verify_password
from app.repositories.user import UserRepository
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: DBSession):
    """Register a new user account and return a JWT access token.

    Args:
        body: Registration details — email, password, and display name.
        db: Injected async database session.

    Returns:
        A ``TokenResponse`` containing the signed JWT for the new user.

    Raises:
        HTTPException: 409 if the email address is already registered.
    """
    repo = UserRepository(db)
    if await repo.get_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = await repo.create(body.email, hash_password(body.password), body.display_name)
    await db.commit()
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DBSession):
    """Authenticate a user and return a JWT access token.

    Args:
        body: Login credentials — email and password.
        db: Injected async database session.

    Returns:
        A ``TokenResponse`` containing the signed JWT.

    Raises:
        HTTPException: 401 if the email is not found or the password is wrong.
    """
    repo = UserRepository(db)
    user = await repo.get_by_email(body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(str(user.id)))


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
