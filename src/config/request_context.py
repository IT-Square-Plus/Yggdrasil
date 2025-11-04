"""
Request context for per-request configuration.

This module provides context variables for storing per-request data like
collection names derived from URL paths, enabling multi-project support
without Docker container proliferation.
"""

from contextvars import ContextVar
from typing import Optional

# Context variable for storing collection name per request
# Usage: /mcp-enigma -> collection_name = "enigma_memories"
_collection_name_ctx: ContextVar[Optional[str]] = ContextVar(
    "collection_name", default=None
)


def set_collection_name(name: str) -> None:
    """
    Set collection name for current request context.

    Args:
        name: Collection name to use for this request
    """
    _collection_name_ctx.set(name)


def get_collection_name() -> Optional[str]:
    """
    Get collection name from current request context.

    Returns:
        Collection name if set, None otherwise
    """
    return _collection_name_ctx.get()


def clear_collection_name() -> None:
    """Clear collection name from current request context."""
    _collection_name_ctx.set(None)


def parse_collection_from_path(path: str, default_suffix: str = "memories") -> str:
    """
    Parse collection name from URL path.

    Examples:
        /mcp-enigma -> "enigma_memories"
        /mcp-alpha -> "alpha_memories"
        /mcp -> "default_memories" (from env var)

    Args:
        path: URL path (e.g., "/mcp-enigma")
        default_suffix: Suffix to append to project name (default: "memories")

    Returns:
        Collection name
    """
    # Remove leading slash and parse
    path = path.lstrip("/")

    # Pattern: mcp-{project_name}
    if path.startswith("mcp-"):
        project_name = path[4:]  # Remove "mcp-" prefix
        if project_name:
            # Sanitize project name (only alphanumeric, underscore, hyphen)
            project_name = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in project_name
            )
            return f"{project_name}_{default_suffix}"

    # Fallback: return None to use default from env var
    return None
