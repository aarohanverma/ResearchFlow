"""Azure Blob Storage backend — used in production."""

from azure.storage.blob.aio import BlobServiceClient

from app.adapters.blob import BlobStorageBackend
from app.core.config import settings

class AzureBlobStorage(BlobStorageBackend):
    """Blob storage backend backed by Azure Blob Storage.

    The container name is read from ``settings.azure_storage_container``
    (env ``AZURE_STORAGEself._container``, default ``researchflow``) so
    operators can deploy multiple environments against a single storage
    account without code changes.
    """

    def __init__(self) -> None:
        """Initialise the Azure BlobServiceClient from the connection string.

        Uses ``settings.azure_storage_connection_string`` to authenticate
        and caches the resolved container name from
        ``settings.azure_storage_container``.
        """
        self._client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        self._container = settings.azure_storage_container

    async def upload(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload bytes to the configured Azure Blob Storage container.

        Args:
            path: Blob path within the container (used as the blob name).
            data: Raw bytes to upload.
            content_type: MIME type of the blob content.

        Returns:
            The canonical blob path (same as ``path``).
        """
        async with self._client:
            container = self._client.get_container_client(self._container)
            blob = container.get_blob_client(path)
            await blob.upload_blob(data, overwrite=True, content_settings={"content_type": content_type})
        return path

    async def download(self, path: str) -> bytes:
        """Download and return the full content of a blob.

        Args:
            path: Blob path within the configured container.

        Returns:
            Raw bytes of the blob content.
        """
        async with self._client:
            blob = self._client.get_blob_client(self._container, path)
            stream = await blob.download_blob()
            return await stream.readall()

    async def exists(self, path: str) -> bool:
        """Check whether a blob exists in the container.

        Args:
            path: Blob path within the configured container.

        Returns:
            ``True`` if the blob exists, ``False`` otherwise.
        """
        async with self._client:
            blob = self._client.get_blob_client(self._container, path)
            return await blob.exists()

    def public_url(self, path: str) -> str:
        """Return the public HTTPS URL for a blob in Azure Blob Storage.

        Args:
            path: Blob path within the configured container.

        Returns:
            A URL of the form
            ``https://<account>.blob.core.windows.net/<container>/<path>``.
        """
        account = self._client.account_name
        return f"https://{account}.blob.core.windows.net/{self._container}/{path}"
