"""UserRepository — users, provider settings, interest profiles, annotations."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.graph import NamespaceSubscription
from app.models.user import Annotation, User, UserInterestProfile, UserProviderSettings


class UserRepository:
    """Data-access layer for ``User``, ``UserInterestProfile``, ``UserProviderSettings``, and ``Annotation`` models.

    Handles all user-related database operations including account creation,
    namespace subscriptions, interest profiles, provider settings, and
    text annotations.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the repository with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for all queries.
        """
        self._db = db

    async def get_by_id(self, user_id: UUID) -> User | None:
        """Fetch a user by primary key.

        Args:
            user_id: The UUID primary key of the user.

        Returns:
            The matching ``User`` ORM object, or ``None`` if not found.
        """
        result = await self._db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Fetch a user by email address (case-insensitive).

        Args:
            email: The email address to look up. Compared after lowercasing.

        Returns:
            The matching ``User`` ORM object, or ``None`` if not found.
        """
        result = await self._db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def create(self, email: str, hashed_password: str, display_name: str) -> User:
        """Create a new user with default provider settings and an empty interest profile.

        The email is normalised to lowercase before storage. A
        ``UserProviderSettings`` row and a ``UserInterestProfile`` row are
        created automatically as part of the same flush.

        Args:
            email: The user's email address (will be lowercased).
            hashed_password: Pre-hashed password string (e.g. bcrypt hash).
            display_name: Human-readable name shown in the UI.

        Returns:
            The newly created ``User`` ORM object with its generated ID.
        """
        user = User(
            email=email.lower(),
            hashed_password=hashed_password,
            display_name=display_name,
        )
        self._db.add(user)
        await self._db.flush()
        # Create default provider settings
        settings_obj = UserProviderSettings(user_id=user.id)
        self._db.add(settings_obj)
        # Create empty interest profile
        profile = UserInterestProfile(user_id=user.id)
        self._db.add(profile)
        await self._db.flush()
        return user

    async def get_namespace_subscriptions(self, user_id: UUID) -> list[str]:
        """Return the namespace keys the user is subscribed to.

        Args:
            user_id: UUID of the user.

        Returns:
            A list of arXiv-style namespace key strings (e.g. ``["cs.AI", "cs.LG"]``).
        """
        result = await self._db.execute(
            select(NamespaceSubscription.namespace_key).where(
                NamespaceSubscription.user_id == user_id
            )
        )
        return [row[0] for row in result.fetchall()]

    async def set_namespace_subscriptions(self, user_id: UUID, namespace_keys: list[str]) -> None:
        """Replace all namespace subscriptions for a user with a new list.

        Deletes every existing ``NamespaceSubscription`` row for the user,
        then inserts one row per key in ``namespace_keys``. The operation is
        flushed but not committed — the caller is responsible for the commit.

        Args:
            user_id: UUID of the user whose subscriptions to replace.
            namespace_keys: The new, complete list of arXiv-style namespace
                keys to subscribe the user to (e.g. ``["cs.AI", "cs.LG"]``).
                Pass an empty list to remove all subscriptions.
        """
        existing = await self._db.execute(
            select(NamespaceSubscription).where(NamespaceSubscription.user_id == user_id)
        )
        for sub in existing.scalars():
            await self._db.delete(sub)

        for ns in namespace_keys:
            self._db.add(NamespaceSubscription(user_id=user_id, namespace_key=ns))
        await self._db.flush()

    async def get_interest_profile(self, user_id: UUID) -> UserInterestProfile | None:
        """Return the interest profile for a user.

        Args:
            user_id: UUID of the user.

        Returns:
            The ``UserInterestProfile`` ORM object, or ``None`` if not yet
            created for this user.
        """
        result = await self._db.execute(
            select(UserInterestProfile).where(UserInterestProfile.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def update_interest_profile(self, user_id: UUID, hot: list[str], cold: list[str]) -> None:
        """Replace the hot and cold subtopic lists on a user's interest profile.

        Does nothing if no profile row exists for the user.

        Args:
            user_id: UUID of the user whose profile to update.
            hot: List of subtopic keys the user wants to see more of.
            cold: List of subtopic keys the user wants to see less of.
        """
        profile = await self.get_interest_profile(user_id)
        if profile:
            profile.hot_subtopics = hot
            profile.cold_subtopics = cold
            await self._db.flush()

    async def get_provider_settings(self, user_id: UUID) -> UserProviderSettings | None:
        """Return the LLM/embedding provider settings for a user.

        Args:
            user_id: UUID of the user.

        Returns:
            The ``UserProviderSettings`` ORM object, or ``None`` if not found.
        """
        result = await self._db.execute(
            select(UserProviderSettings).where(UserProviderSettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def update_provider_settings(self, user_id: UUID, updates: dict) -> None:
        """Patch provider settings for a user.

        Applies each key-value pair in ``updates`` to the existing
        ``UserProviderSettings`` row. Does nothing if no settings row exists.

        Args:
            user_id: UUID of the user whose settings to update.
            updates: Dictionary of column-name → value pairs to apply.
        """
        settings = await self.get_provider_settings(user_id)
        if settings:
            for k, v in updates.items():
                setattr(settings, k, v)
            await self._db.flush()

    async def get_annotations(self, user_id: UUID, paper_id: UUID | None = None) -> list[Annotation]:
        """Return text annotations for a user, optionally scoped to a single paper.

        Args:
            user_id: UUID of the user whose annotations to retrieve.
            paper_id: If provided, only annotations for this paper are returned.
                Defaults to ``None`` (return all annotations for the user).

        Returns:
            A list of ``Annotation`` ORM objects.
        """
        query = select(Annotation).where(Annotation.user_id == user_id)
        if paper_id:
            query = query.where(Annotation.paper_id == paper_id)
        result = await self._db.execute(query)
        return list(result.scalars())

    async def add_annotation(self, user_id: UUID, paper_id: UUID, text: str, note: str | None = None) -> Annotation:
        """Create a new text annotation on a paper for a user.

        Args:
            user_id: UUID of the user creating the annotation.
            paper_id: UUID of the paper being annotated.
            text: The highlighted text passage from the paper.
            note: Optional user-supplied comment on the highlighted text.
                Defaults to ``None``.

        Returns:
            The newly created ``Annotation`` ORM object with its generated ID.
        """
        ann = Annotation(user_id=user_id, paper_id=paper_id, highlighted_text=text, note=note)
        self._db.add(ann)
        await self._db.flush()
        return ann
