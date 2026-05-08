"""Blob storage — local filesystem (dev) or Azure Blob Storage (cloud)."""

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import settings


class BlobStorageBackend(ABC):
    """Abstract base class for blob storage backends.

    Concrete implementations must provide ``upload``, ``download``,
    ``exists``, and ``public_url``.  The local filesystem implementation
    is used in development; Azure Blob Storage is used in production.
    """

    @abstractmethod
    async def upload(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload bytes to blob storage. Returns the canonical path."""

    @abstractmethod
    async def download(self, path: str) -> bytes:
        """Download bytes by canonical path."""

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check whether a blob exists."""

    @abstractmethod
    def public_url(self, path: str) -> str:
        """Return a URL usable by the frontend (may be a signed URL)."""


class LocalBlobStorage(BlobStorageBackend):
    """Blob storage backed by the local filesystem.

    Used in development. Files are written under ``settings.blob_local_dir``
    and served via the FastAPI ``/blobs`` static mount.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        """Initialise the local storage backend.

        Args:
            base_dir: Root directory for blob files. Falls back to
                ``settings.blob_local_dir`` if not provided. The directory
                is created if it does not already exist.
        """
        self._base = Path(base_dir or settings.blob_local_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    async def upload(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Write bytes to the local filesystem at the given relative path.

        Args:
            path: Relative path within the base directory.
            data: Raw bytes to write.
            content_type: MIME type (unused locally; stored for interface
                compatibility).

        Returns:
            The canonical path that was written (same as ``path``).
        """
        import aiofiles
        dest = self._base / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(dest, "wb") as f:
            await f.write(data)
        return path

    async def download(self, path: str) -> bytes:
        """Read and return the bytes stored at the given relative path.

        Args:
            path: Relative path within the base directory.

        Returns:
            Raw bytes of the file.
        """
        import aiofiles
        async with aiofiles.open(self._base / path, "rb") as f:
            return await f.read()

    async def exists(self, path: str) -> bool:
        """Check whether a file exists at the given relative path.

        Args:
            path: Relative path within the base directory.

        Returns:
            ``True`` if the file exists, ``False`` otherwise.
        """
        return (self._base / path).exists()

    def public_url(self, path: str) -> str:
        """Return the URL path served by the FastAPI ``/blobs`` static mount.

        Args:
            path: Relative path within the base directory.

        Returns:
            A URL string of the form ``/blobs/<path>``.
        """
        # Served by FastAPI /static mount in local dev
        return f"/blobs/{path}"


def get_blob_storage() -> BlobStorageBackend:
    """Instantiate and return the configured blob storage backend.

    Returns:
        An ``AzureBlobStorage`` instance when ``settings.blob_backend`` is
        ``"azure"``; otherwise a ``LocalBlobStorage`` instance.
    """
    if settings.blob_backend == "azure":
        from app.adapters.blob.azure_blob import AzureBlobStorage
        return AzureBlobStorage()
    return LocalBlobStorage()


__all__ = ["BlobStorageBackend", "LocalBlobStorage", "get_blob_storage"]
