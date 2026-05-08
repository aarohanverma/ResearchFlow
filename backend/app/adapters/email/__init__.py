"""Email adapter — Resend (cloud-agnostic, same key locally and in Azure)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class EmailMessage:
    """A transactional email message.

    Attributes:
        to: List of recipient email address strings.
        subject: Email subject line.
        html: HTML body of the email.
        text: Optional plain-text fallback body.
    """

    to: list[str]
    subject: str
    html: str
    text: str | None = None


class EmailAdapter(ABC):
    """Abstract base class for email sending backends."""

    @abstractmethod
    async def send(self, message: EmailMessage) -> bool:
        """Send email. Returns True on success."""


class ResendAdapter(EmailAdapter):
    """Email adapter backed by the Resend API.

    Works identically in local development and on Azure — only the API key
    differs. Uses the ``resend`` Python SDK.
    """

    async def send(self, message: EmailMessage) -> bool:
        """Send an email via the Resend API.

        Args:
            message: The ``EmailMessage`` to send.

        Returns:
            ``True`` if the API call succeeded, ``False`` on any exception.
        """
        import resend
        resend.api_key = settings.resend_api_key
        try:
            resend.Emails.send({
                "from": f"{settings.email_from_name} <{settings.email_from}>",
                "to": message.to,
                "subject": message.subject,
                "html": message.html,
                "text": message.text,
            })
            return True
        except Exception:
            return False


def get_email_adapter() -> EmailAdapter:
    """Return the configured email adapter.

    Returns:
        A ``ResendAdapter`` instance ready to send transactional emails.
    """
    return ResendAdapter()


__all__ = ["EmailAdapter", "EmailMessage", "get_email_adapter"]
