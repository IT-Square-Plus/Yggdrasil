"""Yggdrasil MCP Memory Server"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Read version from pyproject.toml (single source of truth)
    __version__ = version("yggdrasil")
except PackageNotFoundError:
    # Fallback during development before package is installed
    __version__ = "dev"
