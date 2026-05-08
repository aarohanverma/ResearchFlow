"""EmailService — Paper of the Day, Weekly Digest, Breakthrough Alert."""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.email import EmailMessage, get_email_adapter
from app.models.paper import Paper, PaperOfDay
from app.models.user import User

log = logging.getLogger(__name__)


class EmailService:
    """Sends transactional emails for Paper of the Day, breakthrough alerts, and weekly digests.

    Delegates delivery to the configured email adapter (injected via
    ``get_email_adapter()``). Each send method respects the per-user
    notification preference flags before dispatching.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the service with a database session and the email adapter.

        Args:
            db: An SQLAlchemy ``AsyncSession`` (currently reserved for future
                per-send logging; not yet used directly in this class).
        """
        self._db = db
        self._adapter = get_email_adapter()

    async def send_potd(self, user: User, paper: Paper, namespace_key: str) -> bool:
        """Send the Paper of the Day email to a single user.

        Skips delivery if the user has ``notify_potd`` disabled.

        Args:
            user: The ``User`` ORM object to notify.
            paper: The ``Paper`` selected as paper of the day.
            namespace_key: The arXiv-style namespace key (e.g. ``"cs.AI"``)
                displayed in the email body.

        Returns:
            ``True`` if the email was sent successfully, ``False`` if the user
            has notifications disabled or delivery failed.
        """
        if not user.notify_potd:
            return False
        html = self._potd_html(paper, namespace_key)
        msg = EmailMessage(
            to=[user.email],
            subject=f"📄 Paper of the Day — {paper.title[:60]}",
            html=html,
        )
        return await self._adapter.send(msg)

    async def send_breakthrough_alert(self, users: list[User], paper: Paper) -> None:
        """Send a breakthrough alert email to each user who has opted in.

        Iterates over the provided user list and sends an alert email for any
        user whose ``notify_breakthrough`` flag is enabled. Delivery failures
        for individual users are not surfaced — the loop continues regardless.

        Args:
            users: List of ``User`` ORM objects to consider for notification.
            paper: The breakthrough ``Paper`` to feature in the alert.
        """
        for user in users:
            if not user.notify_breakthrough:
                continue
            html = self._breakthrough_html(paper)
            await self._adapter.send(EmailMessage(
                to=[user.email],
                subject=f"⚡ Breakthrough Paper — {paper.title[:60]}",
                html=html,
            ))

    async def send_weekly_digest(self, user: User, papers: list[Paper]) -> bool:
        """Send the weekly research digest email to a single user.

        Skips delivery if the user has ``notify_digest`` disabled. The email
        renders up to the first ten papers as a linked list with novelty scores.

        Args:
            user: The ``User`` ORM object to notify.
            papers: The list of ``Paper`` objects to feature in the digest.
                Only the first ten are included in the rendered HTML.

        Returns:
            ``True`` if the email was sent successfully, ``False`` if the user
            has notifications disabled or delivery failed.
        """
        if not user.notify_digest:
            return False
        html = self._digest_html(papers)
        return await self._adapter.send(EmailMessage(
            to=[user.email],
            subject="📊 Your Weekly Research Digest",
            html=html,
        ))

    def _potd_html(self, paper: Paper, ns: str) -> str:
        """Render the Paper-of-the-Day HTML email body."""
        return f"""
        <div style="font-family:sans-serif;max-width:600px;margin:auto">
          <h2>Paper of the Day</h2>
          <p><strong>Namespace:</strong> {ns}</p>
          <h3>{paper.title}</h3>
          <p><em>{', '.join(paper.authors[:3])}</em></p>
          <p>{paper.abstract[:500]}...</p>
          <p>{paper.implications or ''}</p>
          <a href="{paper.source_url}" style="background:#4338ca;color:white;padding:10px 20px;border-radius:6px;text-decoration:none">
            Read Paper →
          </a>
        </div>
        """

    def _breakthrough_html(self, paper: Paper) -> str:
        """Render the breakthrough-alert HTML email body."""
        return f"""
        <div style="font-family:sans-serif;max-width:600px;margin:auto">
          <h2>⚡ Breakthrough Alert</h2>
          <h3>{paper.title}</h3>
          <p><em>{', '.join(paper.authors[:3])}</em></p>
          <p>Novelty score: {paper.novelty_score:.2f} | This paper may represent a significant advance.</p>
          <p>{paper.abstract[:500]}...</p>
          <a href="{paper.source_url}" style="background:#dc2626;color:white;padding:10px 20px;border-radius:6px;text-decoration:none">
            Read Now →
          </a>
        </div>
        """

    def _digest_html(self, papers: list[Paper]) -> str:
        """Render the weekly research-digest HTML email body."""
        items = "".join(
            f"<li><a href='{p.source_url}'>{p.title}</a> — {p.namespace_key} "
            f"(novelty: {p.novelty_score:.1f})</li>"
            for p in papers[:10]
        )
        return f"""
        <div style="font-family:sans-serif;max-width:600px;margin:auto">
          <h2>Your Weekly Research Digest</h2>
          <ul>{items}</ul>
        </div>
        """
