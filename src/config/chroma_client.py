"""
Chroma Cloud Client Configuration
Context7 Best Practice: CloudClient is the recommended way to connect to Chroma Cloud
"""

import logging
from functools import lru_cache

from chromadb import ClientAPI, CloudClient
from chromadb.api.models.Collection import Collection

from .. import __version__
from .settings import get_settings

logger = logging.getLogger("yggdrasil.config.chroma_client")


@lru_cache(maxsize=1)
def get_chroma_client() -> ClientAPI:
    """
    Get Chroma Cloud client (singleton).

    Context7 Recommendation: CloudClient is the recommended way to connect
    to Chroma Cloud (not HttpClient).

    Returns:
        ClientAPI: Chroma Cloud client instance

    Raises:
        ValueError: If connection fails
    """
    settings = get_settings()

    logger.info("☁️ Initializing Chroma Cloud client...")
    logger.info(f"🆔 Tenant: {settings.chroma_tenant}")
    logger.info(f"🗄️ Database: {settings.chroma_database}")

    try:
        client = CloudClient(
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            api_key=settings.chroma_api_key,
            # Default Chroma Cloud settings
            cloud_host="api.trychroma.com",
            cloud_port=8000,
            enable_ssl=True,
        )

        logger.info("✅ Chroma Cloud client initialized successfully")
        return client

    except Exception as e:
        logger.error(f"❌ Failed to initialize Chroma Cloud client: {e}")
        raise ValueError(f"Chroma Cloud connection failed: {e}") from e


def get_collection(collection_name: str | None = None) -> Collection:
    """
    Get or create a collection in Chroma Cloud.

    This function enables dynamic collection selection per project.
    Each collection represents a separate memory space.

    Collection name resolution order:
    1. Explicit collection_name parameter
    2. Request context (from URL path: /mcp-{project})
    3. Default from settings (CHROMA_COLLECTION env var)

    Args:
        collection_name: Optional collection name. If None, uses request context or default.

    Returns:
        Collection: Chroma collection instance

    Raises:
        ValueError: If collection creation fails
    """
    from .request_context import get_collection_name as get_ctx_collection

    client = get_chroma_client()
    settings = get_settings()

    # Resolution order: parameter -> context -> default
    name = collection_name or get_ctx_collection() or settings.chroma_collection

    logger.info(f"Getting or creating collection: {name}")

    try:
        collection = client.get_or_create_collection(
            name=name,
            metadata={
                "description": "Yggdrasil MCP Memory Storage",
                "version": __version__,
                "created_by": "yggdrasil-mcp-memory",
            },
        )

        logger.info(f"✅ Collection '{name}' ready")
        return collection

    except Exception as e:
        logger.error(f"❌ Failed to get/create collection '{name}': {e}")
        raise ValueError(f"Collection operation failed: {e}") from e


def list_collections() -> list[str]:
    """
    List all collections in the current database.

    Returns:
        list[str]: List of collection names
    """
    client = get_chroma_client()

    try:
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        logger.error(f"❌ Failed to list collections: {e}")
        return []


def delete_collection(collection_name: str) -> bool:
    """
    Delete a collection from Chroma Cloud.

    Args:
        collection_name: Name of collection to delete

    Returns:
        bool: True if successful, False otherwise
    """
    client = get_chroma_client()

    try:
        client.delete_collection(name=collection_name)
        logger.info(f"✅ Collection '{collection_name}' deleted")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete collection '{collection_name}': {e}")
        return False
