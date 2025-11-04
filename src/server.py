"""
Yggdrasil
Simple MCP server using FastMCP with Streamable HTTP
"""

import logging
import os
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

from . import __version__
from .config import get_collection, get_settings
from .services.memory_service import MemoryService
from .utils.time_parser import extract_time_expression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("yggdrasil.server")

# Reduce noise from httpx, chromadb, and uvicorn - only show warnings and errors
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(
    logging.WARNING
)  # Hide access logs (GET /ready)

# Create FastMCP server with stateless HTTP
mcp = FastMCP("yggdrasil", stateless_http=True)

# Global server state tracking
_warmup_complete = False
_warmup_start_time = None
_warmup_error = None
_server_start_time = None  # For uptime calculation in /health endpoint (lazy init)
_onnx_model_loaded = False  # Track if ONNX model has been loaded (for cleaner logs)


# Add custom routes to FastMCP using the underlying Starlette app
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """
    Liveness probe endpoint - checks if the process is alive and responding.

    This endpoint ALWAYS returns 200 OK as long as the process is running.
    It does NOT check if the server is ready to serve requests - use /ready for that.

    Returns:
        200 OK with basic server information

    Use cases:
        - Kubernetes liveness probe
        - Docker health check (basic)
        - Process monitoring
    """
    from starlette.responses import JSONResponse as StarletteJSONResponse
    import time

    # Lazy initialization of server start time on first request
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()

    # Calculate uptime since first request
    uptime = int(time.time() - _server_start_time)

    return StarletteJSONResponse(
        {
            "name": "Yggdrasil",
            "status": "alive",
            "version": __version__,
            "protocol": "Streamable HTTP (2025-03-26)",
            "uptime_seconds": uptime,
        }
    )


@mcp.custom_route("/ready", methods=["GET"])
async def readiness_check(request):
    """
    Readiness probe endpoint - checks if server is ready to accept MCP requests.

    Performs intelligent readiness detection:
    1. Checks ChromaDB connection (heartbeat)
    2. Lists available collections (without creating test collection)
    3. If collections exist: tests ONNX model with first available collection
    4. If no collections: reports ready (connection works, ONNX loads on first use)

    Returns:
        200 OK - Server is fully operational and ready to serve requests
        503 Service Unavailable - Server is starting (ONNX model loading) or unhealthy

    Response includes:
        - ready: boolean indicating readiness status
        - status: "operational" | "starting" | "unhealthy"
        - message: optional human-readable status message
        - collections: list of available collection names
        - total_collections: number of collections

    Use cases:
        - Kubernetes readiness probe (only routes traffic when ready)
        - Docker health check (marks container as healthy/unhealthy)
        - Load balancer health checks
        - MCP client connection validation
    """
    from starlette.responses import JSONResponse as StarletteJSONResponse
    from .config import get_chroma_client
    import asyncio

    try:
        # Phase 1: Check ChromaDB connection
        logger.info("💓 Checking ChromaDB heartbeat...")
        client = get_chroma_client()

        # Heartbeat to verify connection
        client.heartbeat()
        logger.info("💚 ChromaDB healthy!")

        # Phase 2: List available collections (without creating any)
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        logger.info(
            f"🗄️ Collections: {', '.join(collection_names) if collection_names else 'none'}"
        )

        # Phase 3: Test ONNX if we have collections with compatible embedding functions
        if collections:
            # Pick a random collection for ONNX test (better distribution than always first)
            import random

            random_collections = collections.copy()
            random.shuffle(random_collections)

            test_collection = None
            tested_collection_name = None

            for col in random_collections:
                # Try to use this collection for ONNX test
                try:
                    # Check if collection has items to test with
                    count = col.count()
                    if count > 0:
                        test_collection = col
                        tested_collection_name = col.name
                        logger.info(
                            f"🔢 Collection '{tested_collection_name}' has {count} memories"
                        )
                        break
                except Exception:
                    # Skip collections with incompatible embedding functions
                    continue

            if test_collection:
                # Test if ONNX model is loaded by doing quick embedding
                async def test_embedding():
                    test_collection.query(query_texts=["readiness"], n_results=1)

                try:
                    global _onnx_model_loaded

                    # Show loading message only on first download
                    if not _onnx_model_loaded:
                        logger.info("=" * 60)
                        logger.info("⏱️  Chroma DB warming up! Waiting for ONNX embedding model...")
                        logger.info("=" * 60)

                    await asyncio.wait_for(test_embedding(), timeout=2.0)

                    # ONNX is ready! Show success message only on first download
                    if not _onnx_model_loaded:
                        logger.info("=" * 60)
                        logger.info("🤖 ONNX embedding model successfully downloaded!")
                        logger.info("=" * 60)
                        _onnx_model_loaded = True
                    return StarletteJSONResponse(
                        status_code=200,
                        content={
                            "name": "Yggdrasil",
                            "ready": True,
                            "status": "operational",
                            "version": __version__,
                            "mcp_endpoint": "/mcp",
                            "protocol": "Streamable HTTP (2025-03-26)",
                            "collections": collection_names,
                            "total_collections": len(collections),
                            "tested_collection": tested_collection_name,
                            "memory_count": test_collection.count(),
                        },
                    )

                except asyncio.TimeoutError:
                    # Embedding test timed out - model still loading
                    return StarletteJSONResponse(
                        status_code=503,
                        content={
                            "name": "Yggdrasil",
                            "ready": False,
                            "status": "starting",
                            "message": "ONNX embedding model is loading (~79MB, usually takes 60-90s)",
                            "version": __version__,
                            "collections": collection_names,
                            "total_collections": len(collections),
                        },
                    )
                except Exception as e:
                    # Collection has incompatible embedding function - skip test
                    logger.info(
                        f"⚠️ Skipping '{tested_collection_name}' - uses different embedding model"
                    )

            # If no testable collections found or test failed, still report ready
            # (connection works, ONNX will load on first MCP request)
            return StarletteJSONResponse(
                status_code=200,
                content={
                    "name": "Yggdrasil",
                    "ready": True,
                    "status": "operational",
                    "version": __version__,
                    "mcp_endpoint": "/mcp",
                    "protocol": "Streamable HTTP (2025-03-26)",
                    "collections": collection_names,
                    "total_collections": len(collections),
                    "message": "ChromaDB connection healthy. ONNX will load on first collection use.",
                },
            )
        else:
            # No collections yet - connection works, ready to create collections on demand
            return StarletteJSONResponse(
                status_code=200,
                content={
                    "name": "Yggdrasil",
                    "ready": True,
                    "status": "operational",
                    "version": __version__,
                    "mcp_endpoint": "/mcp",
                    "protocol": "Streamable HTTP (2025-03-26)",
                    "collections": [],
                    "total_collections": 0,
                    "message": "ChromaDB connection healthy. ONNX will load on first collection use.",
                },
            )

    except Exception as e:
        logger.error(f"💔 ChromaDB unhealthy: {e}")
        return StarletteJSONResponse(
            status_code=503,
            content={
                "name": "Yggdrasil",
                "ready": False,
                "status": "unhealthy",
                "error": str(e),
            },
        )


@mcp.custom_route("/", methods=["GET"])
async def root(request):
    """Root endpoint."""
    from starlette.responses import JSONResponse as StarletteJSONResponse

    return StarletteJSONResponse(
        {
            "name": "Yggdrasil",
            "version": __version__,
            "status": "operational",
            "mcp_endpoint": "/mcp",
            "protocol": "Streamable HTTP (2025-03-26)",
        }
    )


# Get the streamable HTTP app (has /mcp built-in)
app = mcp.streamable_http_app()


# Middleware for path-based collection routing
@app.middleware("http")
async def collection_routing_middleware(request, call_next):
    """
    Parse collection name from URL path and set in request context.

    URL Pattern Examples:
        /mcp-enigma    -> collection: "enigma_memories", routed to /mcp
        /mcp-alpha      -> collection: "alpha_memories", routed to /mcp
        /mcp           -> collection: default from env var

    This enables multi-project support with a single Docker container.
    """
    from .config import (
        parse_collection_from_path,
        set_collection_name,
        clear_collection_name,
    )

    path = request.url.path

    # Parse collection name from path for MCP requests
    if path.startswith("/mcp"):
        collection_name = parse_collection_from_path(path)
        if collection_name:
            set_collection_name(collection_name)
            logger.info(
                f"🎯 Request routed to collection: {collection_name} (from path: {path})"
            )

            # Rewrite path from /mcp-{project} to /mcp for routing
            if path != "/mcp":
                request.scope["path"] = "/mcp"
        else:
            # No project suffix - use default
            settings = get_settings()
            logger.info(
                f"🎯 Request routed to default collection: {settings.chroma_collection} (from path: {path})"
            )

    try:
        response = await call_next(request)
        return response
    finally:
        # Clean up context after request
        clear_collection_name()


# ============================================================================
# TOOLS - Memory Operations
# ============================================================================


@mcp.tool()
async def save_memory(
    content: str, tags: list[str] | None = None, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Store new information in your personal memory collection for later retrieval.

    Use this tool whenever you want to save important information that you might need later:
    facts, notes, code snippets, decisions, meeting notes, ideas, documentation references,
    or any text-based information worth remembering.

    The memory system uses AI embeddings for semantic search, so you can find memories by
    meaning rather than exact keywords. Tags help organize memories into categories for
    easier filtering and management.

    Args:
        content: The information to remember - can be facts, notes, code snippets, ideas,
                 meeting notes, or any text you want to store
                 Examples: "Python uses 0-based indexing", "Meeting with team on API design"
        tags: Optional categorization labels for organization and filtering
              Examples: ["python", "reference"], ["meeting", "team", "2024"]
        metadata: Optional key-value pairs for additional context
                  Examples: {"source": "documentation", "priority": "high"}

    Returns:
        dict: Confirmation with memory ID and storage details
              Format: {"success": bool, "memory": {...}}

    Examples:
        - save_memory("Python supports list comprehensions", tags=["python", "syntax"])
          → Saves a technical reference with categorization

        - save_memory("Decided to use PostgreSQL for main database",
                      tags=["decision", "architecture"],
                      metadata={"date": "2024-11-02", "project": "backend"})
          → Stores an architectural decision with context

        - save_memory("TODO: Implement rate limiting in API endpoints")
          → Quick note without tags or metadata

    Notes:
        - Memories are automatically assigned unique IDs (UUIDs)
        - Created timestamp is automatically recorded
        - Content is embedded for semantic similarity search
        - Duplicate content is allowed - each save creates a new memory
        - Use tags for easy filtering with search_by_tag()
        - Metadata can store any additional structured information
    """
    logger.info(f"Tool 'save_memory' called: content='{content[:100]}...'")
    service = MemoryService()
    result = service.store_memory(content=content, tags=tags, metadata=metadata)
    return {"success": True, "memory": result}


@mcp.tool()
async def get_memory(memory_id: str) -> dict[str, Any]:
    """
    Retrieve a single specific memory by its unique ID (direct lookup).

    Use this tool when you already have the exact memory ID and want to retrieve
    that specific memory. This is a direct database lookup - fast and precise,
    but requires knowing the exact UUID beforehand.

    Common scenarios:
    - You previously retrieved a memory and stored its ID for later reference
    - Another tool returned a memory ID you want to examine in detail
    - You're following up on a specific memory from previous search results
    - You want to verify a memory still exists before updating/deleting

    NOT for: Finding memories by content or topic (use search_memories() instead)

    Args:
        memory_id: The unique identifier (UUID) of the memory to retrieve
                   Example: "885acd52-8612-43da-bbdc-293db0f0b63d"
                   Obtained from search/list/recall operations

    Returns:
        dict: Complete memory object
              Format: {
                  "success": bool,
                  "memory": {
                      "id": str,
                      "content": str,
                      "metadata": {
                          "tags": list[str],
                          "created_at": float,
                          "updated_at": float,
                          ...
                      }
                  }
              }

    Raises:
        ValueError: If memory with given ID is not found

    Examples:
        - get_memory("885acd52-8612-43da-bbdc-293db0f0b63d")
          → Returns the complete memory object if it exists

    Typical workflow:
        1. Search: results = search_memories("Python")
        2. Note ID: memory_id = results["results"][0]["id"]
        3. Later retrieve: memory = get_memory(memory_id)

    Comparison with other tools:
        - Use get_memory() when: You have exact memory ID
        - Use search_memories() when: Looking for content by topic
        - Use list_memories() when: Browsing without specific ID
        - Use recall_memory() when: Searching within time period

    Notes:
        - This is a direct lookup operation (very fast)
        - No semantic search or ranking involved
        - Returns single memory or raises ValueError
        - ID must be exact UUID from previous operations
        - Useful for verifying memory exists before operations
    """
    logger.info(f"Tool 'get_memory' called: memory_id='{memory_id}'")
    service = MemoryService()
    result = service.get_memory(memory_id=memory_id)
    return {"success": True, "memory": result}


@mcp.tool()
async def search_memories(
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform intelligent semantic search across your memory collection using AI embeddings.

    This is the primary way to find relevant memories when you're looking for specific information.
    Unlike keyword matching, semantic search understands the meaning and context of your query,
    finding memories that are conceptually related even if they don't share exact words.

    Use this tool when you:
    - Want to find memories about a specific topic or concept
    - Need to locate related information without knowing exact wording
    - Are searching for similar ideas, solutions, or references
    - Want relevance-ranked results (most similar first)

    Perfect for questions like: "What did I learn about async programming?", "Find notes
    about database optimization", "Show me API design decisions"

    Args:
        query: Your search query - can be keywords, phrases, questions, or concepts
               The more specific your query, the better the results
               Examples: "Python async patterns", "How to handle rate limiting",
                        "Database indexing strategies", "Meeting notes about API"
        limit: Maximum number of results to return (default: 10)
               Increase for broader results, decrease for top matches only
        filters: Optional ChromaDB metadata filters for precise filtering
                 Example: {"tags": {"$contains": "python"}}

    Returns:
        dict: Search results with memories ordered by relevance
              Format: {
                  "success": bool,
                  "results": [
                      {
                          "id": str,              # Memory UUID
                          "content": str,         # Memory content
                          "metadata": dict,       # Tags, timestamps, etc.
                          "score": float          # Similarity score (higher = more relevant)
                      },
                      ...
                  ],
                  "count": int
              }

    Examples:
        - search_memories("Python async programming")
          → Finds memories about asyncio, coroutines, async/await patterns

        - search_memories("database performance optimization", limit=5)
          → Returns top 5 most relevant memories about DB optimization

        - search_memories("API design decisions from last sprint")
          → Semantic search combines concept matching with time context

    Comparison with other tools:
        - Use search_memories() when: Looking for information by topic/concept
        - Use list_memories() when: Browsing all memories chronologically
        - Use search_by_tag() when: Filtering by specific categories
        - Use recall_memory() when: Searching within a time period
        - Use get_memory() when: You have the exact memory ID

    Notes:
        - Results are ranked by semantic similarity (relevance score)
        - Understands synonyms and related concepts automatically
        - Works across different phrasings of the same idea
        - More specific queries generally yield better results
        - Combines well with time filters via recall_memory()
        - Score indicates how closely the memory matches your query
    """
    logger.info(f"Tool 'search_memories' called: query='{query}', limit={limit}")
    service = MemoryService()
    results = service.search_memories(query=query, limit=limit, filters=filters)
    return {"success": True, "results": results, "count": len(results)}


@mcp.tool()
async def debug_retrieve(
    query: str,
    limit: int = 10,
    similarity_threshold: float = -1.0,
) -> dict[str, Any]:
    """
    Retrieve memories with debug information for semantic search analysis.

    Returns memories with raw similarity scores, distances, and memory IDs to help
    understand search quality and optimize similarity thresholds. Perfect for debugging
    why certain memories are or aren't being returned.

    Note: Similarity scores can be negative when distance > 1.0. The default threshold
    of -1.0 captures most results. Use higher thresholds (e.g., 0.3) for filtering.

    Args:
        query: Search query for debugging retrieval
        limit: Maximum number of results (default: 10)
        similarity_threshold: Minimum similarity score, can be negative (default: -1.0)

    Returns:
        dict: Search results with debug information
              Format: {"success": bool, "results": [...], "count": int}
              Each result includes: id, content, metadata, score, debug_info

    Examples:
        - debug_retrieve("Python programming") - get all results with debug scores
        - debug_retrieve("AI", similarity_threshold=0.3) - filter by minimum similarity
        - debug_retrieve("test", limit=5) - limit debug results

    Debug info includes:
        - raw_similarity: Similarity score (can be negative, higher is better)
        - raw_distance: Distance value from ChromaDB
        - memory_id: Unique identifier for cross-reference
    """
    logger.info(
        f"Tool 'debug_retrieve' called: query='{query}', threshold={similarity_threshold}"
    )

    service = MemoryService()

    try:
        results = service.debug_retrieve(
            query=query, limit=limit, similarity_threshold=similarity_threshold
        )
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"❌ debug_retrieve failed: {e}")
        return {"success": False, "error": str(e), "results": [], "count": 0}


@mcp.tool()
async def list_memories(
    limit: int = 100, offset: int = 0, filters: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Browse all stored memories in chronological order (newest first) without searching.

    This tool is for exploration and overview - use it when you want to see what's stored
    in your memory collection without filtering by topic or keywords. Unlike search_memories()
    which finds specific content by relevance, this returns memories in the order they were
    created (most recent first).

    Use this tool when you:
    - Want to see recent activity and what was saved lately
    - Need an overview of your entire memory collection
    - Are exploring without a specific search goal
    - Want to review all memories in a category (using filters)
    - Are checking what information is available before searching

    NOT for: Finding specific information (use search_memories() instead)

    Args:
        limit: Number of memories to return (default: 100)
               Use lower values (10-20) for quick overviews
               Use higher values (50-100) for comprehensive browsing
        offset: Number of memories to skip for pagination (default: 0)
                Example: offset=100, limit=50 returns memories 101-150
        filters: Optional ChromaDB metadata filters for category browsing
                 Example: {"tags": {"$contains": "work"}} to list only work memories

    Returns:
        dict: List of memories in chronological order
              Format: {
                  "success": bool,
                  "memories": [
                      {
                          "id": str,              # Memory UUID
                          "content": str,         # Memory content
                          "metadata": dict        # Tags, timestamps, created_at, etc.
                      },
                      ...
                  ],
                  "count": int  # Number of memories returned
              }

    Examples:
        - list_memories(limit=10)
          → Shows 10 most recent memories (quick overview)

        - list_memories(limit=50, offset=50)
          → Shows memories 51-100 (pagination through history)

        - list_memories(filters={"tags": {"$contains": "meeting"}})
          → Browse all meeting-related memories chronologically

    Comparison with other tools:
        - Use list_memories() when: Browsing without specific search goal
        - Use search_memories() when: Looking for specific topics/concepts
        - Use search_by_tag() when: Filtering by exact tag matches
        - Use recall_memory() when: Browsing within a time period

    Notes:
        - Results are NOT ranked by relevance (chronological only)
        - Most recently created memories appear first
        - No semantic analysis performed (just retrieval)
        - Faster than semantic search for browsing
        - Good for checking recent activity
        - Use pagination for large collections (offset + limit)
    """
    logger.info(f"Tool 'list_memories' called: limit={limit}, offset={offset}")
    service = MemoryService()
    results = service.list_memories(limit=limit, offset=offset, filters=filters)
    return {"success": True, "memories": results, "count": len(results)}


@mcp.tool()
async def delete_memory(memory_id: str) -> dict[str, Any]:
    """
    Permanently delete a specific memory from your collection using its unique ID.

    WARNING: This action cannot be undone! The memory will be permanently removed
    from the database. Consider creating a backup first if you're unsure.

    Use this tool when you need to remove a single, specific memory that you've
    identified by its ID. For bulk deletions, use delete_by_tag(), delete_by_timeframe(),
    or other deletion tools instead.

    The memory ID can be obtained from:
    - search_memories() results (each memory includes "id" field)
    - list_memories() output (each memory has "id")
    - recall_memory() or recall_by_timeframe() results
    - get_memory() if you already have the ID
    - Any operation that returns memory objects

    Args:
        memory_id: The unique identifier (UUID) of the memory to delete
                   Example: "885acd52-8612-43da-bbdc-293db0f0b63d"
                   This ID is returned by all memory retrieval operations

    Returns:
        dict: Deletion confirmation
              Format: {"success": bool}

    Examples:
        1. First, find the memory to delete:
           results = search_memories("outdated information")
           memory_id = results["results"][0]["id"]

        2. Then delete it:
           delete_memory(memory_id)
           → Permanently removes the memory

        Alternative workflow:
        - list_memories(limit=10)  # Browse recent memories
        - Identify unwanted memory ID
        - delete_memory("abc-123-...")  # Delete it

    Use cases:
        - Removing outdated or incorrect information
        - Cleaning up test/temporary memories
        - Deleting sensitive information
        - Correcting mistakes (delete + save corrected version)

    Comparison with other deletion tools:
        - Use delete_memory() when: Removing one specific memory by ID
        - Use delete_by_tag() when: Removing all memories with certain tags
        - Use delete_by_timeframe() when: Removing memories from a time period
        - Use delete_before_date() when: Implementing retention policies

    Notes:
        - Deletion is immediate and permanent
        - No way to recover deleted memories (make backups!)
        - Memory ID must be exact UUID
        - Returns success=true even if memory doesn't exist
        - For bulk operations, use specialized deletion tools
    """
    logger.info(f"Tool 'delete_memory' called: memory_id='{memory_id}'")
    service = MemoryService()
    success = service.delete_memory(memory_id)
    return {"success": success}


@mcp.tool()
async def get_memory_stats() -> dict[str, Any]:
    """
    Get quick statistics and overview of your memory collection.

    This tool provides a fast summary of your memory collection without retrieving
    actual memory content. Use it to check collection size, verify the service is
    working, or get a quick overview before performing operations.

    Useful for:
    - Checking how many memories are stored
    - Verifying the memory service is operational
    - Quick health check before major operations
    - Monitoring collection growth over time
    - Deciding pagination limits for list_memories()

    Returns:
        dict: Collection statistics
              Format: {
                  "success": bool,
                  "count": int  # Total number of memories in collection
              }

    Examples:
        - get_memory_stats()
          → {"success": true, "count": 1247}
          Shows you have 1,247 memories stored

    Use cases:
        - Before cleanup: "I have 5000 memories, time to clean duplicates"
        - Before backup: "Let me check the count before exporting"
        - Monitoring: "Collection growing too large, need retention policy"
        - Debugging: "Stats show 0 memories, service might have issues"

    Comparison with other tools:
        - Use get_memory_stats() when: Need quick count/overview
        - Use check_health() when: Need comprehensive system health check
        - Use list_memories() when: Want to see actual memory content
        - Use search_memories() when: Looking for specific information

    Notes:
        - Very fast operation (no content retrieval)
        - Does not load or process actual memories
        - Safe to call frequently for monitoring
        - Count includes all memories regardless of tags/filters
        - Useful for deciding if cleanup/optimization is needed
    """
    logger.info("Tool 'get_memory_stats' called")
    service = MemoryService()
    count = service.count_memories()
    return {"success": True, "count": count}


@mcp.tool()
async def check_health() -> dict[str, Any]:
    """
    Perform comprehensive health check on the memory system.

    This tool verifies the operational status of the memory system by checking:
    - ChromaDB connection and collection accessibility
    - Query functionality and embedding capabilities
    - Memory count and collection statistics

    Returns:
        dict: Health status report
            Format: {
                "success": bool,
                "status": "healthy" | "degraded" | "unhealthy",
                "checks": {
                    "chromadb_connection": bool,
                    "collection_accessible": bool,
                    "query_functional": bool
                },
                "stats": {
                    "total_memories": int,
                    "collection_name": str
                },
                "errors": list[str]  # Empty if healthy
            }

    Status meanings:
        - "healthy": All systems operational
        - "degraded": Some non-critical issues detected
        - "unhealthy": Critical failures, system not functional

    Examples:
        - check_health() → Full system diagnostic report
        - Use before critical operations to verify system readiness
        - Use for monitoring and alerting in production

    Notes:
        - Non-invasive check that doesn't modify data
        - Safe to run frequently for monitoring
        - Provides actionable error messages if issues detected
    """
    logger.info("Tool 'check_health' called")
    service = MemoryService()
    health_status = service.check_health()
    return {"success": True, **health_status}


@mcp.tool()
async def recall_memory(query: str, limit: int = 10) -> dict[str, Any]:
    """
    Retrieve memories using natural language time expressions and optional semantic search.

    This tool combines time-based filtering with semantic search to find memories from specific time periods.
    Time expressions are automatically extracted from the query and used to filter results.

    Supported time expressions:
    - Relative: "yesterday", "today", "2 days ago", "3 weeks ago", "1 month ago"
    - Periods: "last week", "last month", "this week", "this year"
    - Months: "january", "february", etc. (uses current or previous year)
    - Time of day: "yesterday morning", "today afternoon", "2 days ago evening"
    - Fuzzy: "recent", "recently", "lately" (defaults to last 7 days)

    Examples:
    - "what did I save yesterday" - finds memories from yesterday
    - "memories about Python from last week" - semantic search within last week
    - "recent memories about AI" - searches last 7 days for AI-related content
    - "what happened this morning" - finds today's morning memories

    Args:
        query: Natural language query with optional time expressions and search terms
        limit: Maximum number of results to return (default: 10)

    Returns:
        dict: Search results with memories matching time period and semantic query
    """
    logger.info(f"Tool 'recall_memory' called: query='{query}', limit={limit}")

    # Extract time expression from query
    cleaned_query, (start_ts, end_ts) = extract_time_expression(query)
    logger.info(f"Extracted: cleaned='{cleaned_query}', start={start_ts}, end={end_ts}")

    service = MemoryService()

    # Build time filter if timestamps were found
    where_clause = None
    if start_ts is not None and end_ts is not None:
        where_clause = {
            "$and": [
                {"created_at": {"$gte": start_ts}},
                {"created_at": {"$lte": end_ts}},
            ]
        }
        logger.info(f"Time filter: {start_ts} to {end_ts}")

    # If there's a cleaned query (semantic part), do semantic search with time filter
    if cleaned_query:
        logger.info(f"Semantic search with query: '{cleaned_query}'")
        results = service.search_memories(
            query=cleaned_query, limit=limit, filters=where_clause
        )
    # If only time expression (no semantic query), list memories in time range
    elif where_clause:
        logger.info("Time-only filtering (no semantic query)")
        results = service.list_memories(limit=limit, filters=where_clause)
    # If no time and no query, fall back to regular search
    else:
        logger.info("No time expression found, falling back to semantic search")
        results = service.search_memories(query=query, limit=limit)

    return {"success": True, "results": results, "count": len(results)}


@mcp.tool()
async def recall_by_timeframe(
    start_time: float, end_time: float, query: str | None = None, limit: int = 10
) -> dict[str, Any]:
    """
    Retrieve memories within a specific timeframe with optional semantic search.

    This is the explicit version of recall_memory - instead of natural language
    time parsing, you provide exact Unix timestamps for start and end times.
    Perfect for programmatic access or when you know exact time boundaries.

    Args:
        start_time: Start timestamp (Unix timestamp as float)
                   Example: 1730505600.0 (November 2, 2024 00:00:00 UTC)
        end_time: End timestamp (Unix timestamp as float)
                 Example: 1730592000.0 (November 3, 2024 00:00:00 UTC)
        query: Optional semantic search query to filter results within timeframe
        limit: Maximum number of results to return (default: 10)

    Returns:
        dict: Search results with memories from the specified timeframe
              Format: {"success": bool, "results": [...], "count": int}

    Examples:
        - recall_by_timeframe(1730505600.0, 1730592000.0)
          → All memories from Nov 2-3, 2024

        - recall_by_timeframe(time.time() - 86400, time.time(), query="Python")
          → Python-related memories from last 24 hours

        - recall_by_timeframe(1730505600.0, 1730592000.0, limit=5)
          → First 5 memories from specified timeframe

    Notes:
        - Both start_time and end_time are inclusive
        - If query is provided, performs semantic search within timeframe
        - If no query, returns all memories in chronological order
        - Uses created_at timestamp for filtering
        - Raises ValueError if start_time > end_time
    """
    logger.info(
        f"Tool 'recall_by_timeframe' called: start={start_time}, end={end_time}, query='{query}', limit={limit}"
    )

    service = MemoryService()

    try:
        results = service.recall_by_timeframe(
            start_time=start_time, end_time=end_time, query=query, limit=limit
        )
        return {"success": True, "results": results, "count": len(results)}
    except ValueError as e:
        logger.error(f"❌ recall_by_timeframe validation error: {e}")
        return {"success": False, "error": str(e), "results": [], "count": 0}
    except Exception as e:
        logger.error(f"❌ recall_by_timeframe failed: {e}")
        return {"success": False, "error": str(e), "results": [], "count": 0}


@mcp.tool()
async def search_by_tag(tags: list[str]) -> dict[str, Any]:
    """
    Find memories by category tags using flexible ANY matching logic.

    Returns memories that contain ANY of the specified tags - perfect for broad searches
    across related categories. If a memory has even one of your search tags, it will be
    included in results. This is the most flexible tag search method.

    Use this tool when you:
    - Want to gather all memories from related categories
    - Need broad coverage across multiple topics
    - Are exploring what's stored under various tags
    - Want to find memories that might be in any of several categories

    ANY logic means: "Show me memories that have tag1 OR tag2 OR tag3"

    Args:
        tags: List of tags to search for. Returns memories containing ANY of these tags.
              Examples: ["work", "meeting"], ["python", "coding"], ["important", "urgent"]
              Even if memory has only one matching tag, it's included in results

    Returns:
        dict: Search results with memories matching any of the tags
              Format: {
                  "success": bool,
                  "memories": [
                      {
                          "id": str,
                          "content": str,
                          "metadata": dict  # Includes tags, timestamps, etc.
                      },
                      ...
                  ],
                  "count": int
              }

    Examples:
        - search_by_tag(["work", "meeting"])
          → Finds ALL memories tagged "work" OR "meeting" (or both)
          Use case: "Show me everything related to work or meetings"

        - search_by_tag(["python", "javascript", "rust"])
          → Finds memories about ANY of these programming languages
          Use case: "Show me all my programming language notes"

        - search_by_tag(["urgent"])
          → Finds all urgent items (single tag search)
          Use case: "What needs immediate attention?"

        - search_by_tag(["reference", "documentation", "tutorial"])
          → Finds educational/reference materials
          Use case: "Show me all my learning resources"

    Real-world scenarios:
        - Project review: search_by_tag(["project-x", "sprint-5", "retrospective"])
        - Learning session: search_by_tag(["tutorial", "guide", "example"])
        - Priority check: search_by_tag(["urgent", "important", "deadline"])
        - Tech stack: search_by_tag(["react", "nodejs", "postgresql"])

    Comparison with other tools:
        - Use search_by_tag() when: Broad search across related categories (OR logic)
        - Use search_by_all_tags() when: Precise match requiring all tags (AND logic)
        - Use search_memories() when: Searching by content/meaning, not categories
        - Use list_memories() when: Browsing chronologically without filters

    Notes:
        - ANY logic = inclusive search (casts wide net)
        - Memory with "work" tag matches search_by_tag(["work", "personal"])
        - More tags in search = broader results
        - Results NOT ranked (unlike semantic search)
        - Fast category-based filtering
        - Useful for exploratory browsing by topic
    """
    logger.info(f"Tool 'search_by_tag' called: tags={tags}")

    if not tags:
        return {
            "success": False,
            "error": "Tags are required",
            "memories": [],
            "count": 0,
        }

    service = MemoryService()

    try:
        memories = service.search_by_tags(tags)
        return {"success": True, "memories": memories, "count": len(memories)}
    except Exception as e:
        logger.error(f"❌ search_by_tag failed: {e}")
        return {"success": False, "error": str(e), "memories": [], "count": 0}


@mcp.tool()
async def delete_by_tag(tags: list[str]) -> dict[str, Any]:
    """
    Delete all memories containing any of the specified tags.

    WARNING: This permanently deletes memories and cannot be undone. Use with caution.
    Uses ANY matching logic - deletes memories with at least one of the specified tags.

    Args:
        tags: List of tags to delete. Memories containing ANY of these tags will be permanently deleted.
              Examples: ["temporary", "outdated"], ["test", "draft"]

    Returns:
        dict: Deletion results with count and matched tags
              Format: {"success": bool, "deleted_count": int, "matched_tags": [...], "message": str}

    Examples:
        - delete_by_tag(["test"]) - deletes all test memories
        - delete_by_tag(["temporary", "draft"]) - deletes memories with "temporary" OR "draft"
    """
    logger.info(f"Tool 'delete_by_tag' called: tags={tags}")

    if not tags:
        return {"success": False, "deleted_count": 0, "error": "Tags are required"}

    service = MemoryService()

    try:
        result = service.delete_by_tags(tags)
        return result
    except Exception as e:
        logger.error(f"❌ delete_by_tag failed: {e}")
        return {"success": False, "deleted_count": 0, "error": str(e)}


@mcp.tool()
async def search_by_all_tags(tags: list[str]) -> dict[str, Any]:
    """
    Find memories that have ALL specified tags (precise filtering with AND logic).

    Returns ONLY memories that contain EVERY one of the specified tags - perfect for
    narrow, precise searches when you need specific combinations. If a memory is missing
    even one of your search tags, it won't be included. This is the most restrictive
    tag search method.

    Use this tool when you:
    - Need precise filtering with multiple required criteria
    - Want to find intersection of multiple categories
    - Are looking for specific combinations of attributes
    - Need to narrow down results to exact matches

    ALL logic means: "Show me memories that have tag1 AND tag2 AND tag3"

    Args:
        tags: List of tags. Returns memories containing ALL of these tags.
              Examples: ["python", "test"], ["work", "urgent"]
              Memory must have EVERY tag in the list to be included

    Returns:
        dict: Search results with memories matching all tags
              Format: {
                  "success": bool,
                  "memories": [
                      {
                          "id": str,
                          "content": str,
                          "metadata": dict  # Must contain ALL search tags
                      },
                      ...
                  ],
                  "count": int
              }

    Examples:
        - search_by_all_tags(["python", "test"])
          → Finds ONLY memories with BOTH "python" AND "test" tags
          Use case: "Show me Python code that has tests"

        - search_by_all_tags(["work", "urgent", "customer"])
          → Finds ONLY memories tagged with ALL three
          Use case: "Critical customer work items only"

        - search_by_all_tags(["tutorial", "advanced", "python"])
          → Finds advanced Python tutorials specifically
          Use case: "I need advanced-level Python learning materials"

        - search_by_all_tags(["2024", "Q1", "review"])
          → Finds Q1 2024 review items specifically
          Use case: "Show me first quarter review notes"

    Real-world scenarios:
        - Precise project filtering: search_by_all_tags(["project-x", "backend", "bug"])
          → ONLY backend bugs for project-x (excludes frontend bugs, other projects)

        - Skill intersection: search_by_all_tags(["react", "typescript", "production"])
          → Production React+TypeScript code specifically

        - Meeting notes: search_by_all_tags(["meeting", "architecture", "Q4-2024"])
          → Architecture meeting notes from Q4 2024 specifically

        - Priority items: search_by_all_tags(["high-priority", "technical-debt", "backend"])
          → High-priority backend technical debt specifically

    Comparison with other tools:
        - Use search_by_all_tags() when: Need precise intersection (AND logic)
        - Use search_by_tag() when: Want broad coverage (OR logic)
        - Use search_memories() when: Searching by content/meaning
        - Use list_memories() when: Browsing chronologically

    Comparison: ANY vs ALL logic:
        Scenario: Memory tagged ["python", "test", "tutorial"]
        - search_by_tag(["python", "javascript"]) → MATCH (has python)
        - search_by_all_tags(["python", "javascript"]) → NO MATCH (missing javascript)
        - search_by_all_tags(["python", "test"]) → MATCH (has both)

    Notes:
        - ALL logic = restrictive search (narrow results)
        - More tags in search = fewer results (stricter criteria)
        - Memory must have EVERY search tag to match
        - Results NOT ranked (unlike semantic search)
        - Perfect for multi-criteria filtering
        - Use when you need exact combination of tags
    """
    logger.info(f"Tool 'search_by_all_tags' called: tags={tags}")

    if not tags:
        return {
            "success": False,
            "error": "Tags are required",
            "memories": [],
            "count": 0,
        }

    service = MemoryService()

    try:
        memories = service.search_by_all_tags(tags)
        return {"success": True, "memories": memories, "count": len(memories)}
    except Exception as e:
        logger.error(f"❌ search_by_all_tags failed: {e}")
        return {"success": False, "error": str(e), "memories": [], "count": 0}


@mcp.tool()
async def delete_by_all_tags(tags: list[str]) -> dict[str, Any]:
    """
    Delete memories that contain ALL of the specified tags.

    WARNING: This permanently deletes memories and cannot be undone. Use with caution.
    Uses ALL matching logic - deletes memories only if they contain EVERY specified tag.

    This is different from delete_by_tag which uses ANY logic. Use delete_by_all_tags
    when you need precise targeting of memories that have a specific combination of tags.

    Args:
        tags: List of tags. Memories containing ALL of these tags will be permanently deleted.
              Examples: ["python", "test"], ["work", "project1"]

    Returns:
        dict: Deletion results with count and matched tags
              Format: {"success": bool, "deleted_count": int, "matched_tags": [...], "message": str}

    Examples:
        - delete_by_all_tags(["python", "test"]) - deletes ONLY memories with BOTH "python" AND "test"
        - delete_by_all_tags(["work", "archived"]) - deletes ONLY memories with BOTH "work" AND "archived"

    Comparison:
        - delete_by_tag(["A", "B"]) → deletes memories with A OR B
        - delete_by_all_tags(["A", "B"]) → deletes memories with A AND B
    """
    logger.info(f"Tool 'delete_by_all_tags' called: tags={tags}")

    if not tags:
        return {"success": False, "deleted_count": 0, "error": "Tags are required"}

    service = MemoryService()

    try:
        result = service.delete_by_all_tags(tags)
        return result
    except Exception as e:
        logger.error(f"❌ delete_by_all_tags failed: {e}")
        return {"success": False, "deleted_count": 0, "error": str(e)}


@mcp.tool()
async def delete_by_timeframe(start_time: float, end_time: float) -> dict[str, Any]:
    """
    Delete memories within a specific timeframe.

    WARNING: This permanently deletes memories and cannot be undone. Use with caution.

    Deletes all memories where created_at is between start_time and end_time (inclusive).
    Useful for cleaning up test data, removing old memories from a specific period,
    or managing memory retention policies.

    Args:
        start_time: Start timestamp (Unix timestamp as float)
                   Example: 1730505600.0 (November 2, 2024 00:00:00 UTC)
        end_time: End timestamp (Unix timestamp as float)
                 Example: 1730592000.0 (November 3, 2024 00:00:00 UTC)

    Returns:
        dict: Deletion results
              Format: {
                  "success": bool,
                  "deleted_count": int,
                  "timeframe": {
                      "start": float,
                      "end": float,
                      "start_iso": str,
                      "end_iso": str
                  },
                  "message": str
              }

    Raises:
        ValueError: If start_time > end_time

    Examples:
        - delete_by_timeframe(1730505600.0, 1730592000.0)
          → Deletes all memories created between Nov 2-3, 2024

        - delete_by_timeframe(time.time() - 86400, time.time())
          → Deletes all memories from last 24 hours

    Notes:
        - Uses created_at timestamp for filtering
        - Both start_time and end_time are inclusive
        - Returns ISO-formatted dates for human readability
        - Safe to run even if no memories match the timeframe
    """
    logger.info(
        f"Tool 'delete_by_timeframe' called: start={start_time}, end={end_time}"
    )

    service = MemoryService()

    try:
        result = service.delete_by_timeframe(start_time, end_time)
        return result
    except Exception as e:
        logger.error(f"❌ delete_by_timeframe failed: {e}")
        return {"success": False, "deleted_count": 0, "error": str(e)}


@mcp.tool()
async def delete_before_date(timestamp: float) -> dict[str, Any]:
    """
    Delete all memories created before a specific date.

    WARNING: This permanently deletes memories and cannot be undone. Use with caution.

    Deletes all memories where created_at < timestamp. Useful for implementing
    data retention policies, cleaning up old memories, or preparing for fresh starts.

    Args:
        timestamp: Cutoff timestamp (Unix timestamp as float)
                  All memories created before this time will be deleted
                  Example: 1730505600.0 (November 2, 2024 00:00:00 UTC)

    Returns:
        dict: Deletion results
              Format: {
                  "success": bool,
                  "deleted_count": int,
                  "cutoff_date": {
                      "timestamp": float,
                      "iso": str
                  },
                  "message": str
              }

    Examples:
        - delete_before_date(1730505600.0)
          → Deletes all memories created before November 2, 2024

        - delete_before_date(time.time() - 2592000)
          → Deletes all memories older than 30 days

    Notes:
        - Uses created_at timestamp for filtering
        - Cutoff is exclusive (memories created exactly at timestamp are kept)
        - Returns ISO-formatted date for human readability
        - Safe to run even if no memories match the criteria
        - Useful for implementing "delete after X days" policies
    """
    logger.info(f"Tool 'delete_before_date' called: timestamp={timestamp}")

    service = MemoryService()

    try:
        result = service.delete_before_date(timestamp)
        return result
    except Exception as e:
        logger.error(f"❌ delete_before_date failed: {e}")
        return {"success": False, "deleted_count": 0, "error": str(e)}


@mcp.tool()
async def update_memory(
    memory_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Update an existing memory's content, tags, and/or metadata.

    This is the full update operation - use this when you need to change the memory content
    or update multiple fields at once. For tag-only updates, consider using update_memory_tags()
    which is more efficient.

    Args:
        memory_id: The unique identifier of the memory to update
        content: New content for the memory (optional - keeps existing if not provided)
        tags: New list of tags (optional - keeps existing if not provided)
        metadata: Additional metadata to update (optional - merged with existing)

    Returns:
        dict: Updated memory object with id, content, and metadata

    Raises:
        ValueError: If memory with given ID is not found

    Examples:
        - update_memory("abc-123", content="Updated content")
          → Updates only the content, keeps existing tags and metadata

        - update_memory("abc-123", tags=["python", "updated"])
          → Updates only the tags, keeps existing content and metadata

        - update_memory("abc-123", content="New content", tags=["new", "tags"])
          → Updates both content and tags

        - update_memory("abc-123", metadata={"priority": "high"})
          → Adds/updates metadata field, keeps existing content and tags

    Notes:
        - updated_at timestamp is automatically updated
        - Partial updates are supported - only provided fields are changed
        - Metadata is merged with existing (doesn't replace all metadata)
        - For tag-only updates, use update_memory_tags() for better control
    """
    logger.info(f"Tool 'update_memory' called: memory_id='{memory_id}'")
    service = MemoryService()
    result = service.update_memory(
        memory_id=memory_id, content=content, tags=tags, metadata=metadata
    )
    return {"success": True, "memory": result}


@mcp.tool()
async def update_memory_tags(
    memory_id: str, tags: list[str], mode: str = "replace"
) -> dict[str, Any]:
    """
    Update tags for an existing memory without modifying its content.

    This is a specialized tool for tag management - more efficient and safer than
    update_memory() when you only want to change tags. Supports three modes:
    replace, append, and remove.

    Args:
        memory_id: The unique identifier of the memory to update
        tags: List of tags to apply
        mode: Update mode (default: "replace")
              - "replace": Replace all existing tags with new ones
              - "append": Add new tags to existing tags (no duplicates)
              - "remove": Remove specified tags from existing tags

    Returns:
        dict: Update results with operation details
              Format: {
                  "success": bool,
                  "id": str,
                  "operation": str,
                  "previous_tags": list[str],
                  "new_tags": list[str],
                  "metadata": dict
              }

    Examples:
        - update_memory_tags("abc-123", ["python", "coding"], "replace")
          → Replaces all tags with ["python", "coding"]

        - update_memory_tags("abc-123", ["important"], "append")
          → Adds "important" to existing tags (if not already present)

        - update_memory_tags("abc-123", ["draft", "temp"], "remove")
          → Removes "draft" and "temp" tags from existing tags
    """
    logger.info(
        f"Tool 'update_memory_tags' called: memory_id={memory_id}, tags={tags}, mode={mode}"
    )

    if not memory_id:
        return {"success": False, "error": "memory_id is required"}

    if not tags:
        return {"success": False, "error": "tags list cannot be empty"}

    service = MemoryService()

    try:
        result = service.update_memory_tags(memory_id, tags, mode)
        return result
    except Exception as e:
        logger.error(f"❌ update_memory_tags failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def update_memory_metadata(
    memory_id: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    """
    Update only the metadata of an existing memory without changing its content.

    This is a specialized update operation - unlike update_memory() which can change
    content, tags, and metadata, this method ONLY updates metadata fields. The memory
    content and tags remain unchanged. Perfect for adding annotations, status updates,
    or tracking information without touching the core memory content.

    Args:
        memory_id: The unique identifier of the memory to update
        metadata: New metadata fields to add or update (merged with existing)
                 Example: {"priority": "high", "project": "yggdrasil"}

    Returns:
        dict: Update results with confirmation
              Format: {
                  "success": bool,
                  "id": str,
                  "operation": "metadata_update",
                  "updated_fields": list[str],
                  "metadata": dict  # Full updated metadata
              }

    Examples:
        - update_memory_metadata("abc-123", {"priority": "high"})
          → Adds/updates priority field, keeps all other metadata

        - update_memory_metadata("abc-123", {"reviewed": True, "reviewer": "Claude"})
          → Adds review metadata without touching content

        - update_memory_metadata("abc-123", {"status": "completed", "completion_date": "2024-11-02"})
          → Updates status tracking metadata

    Notes:
        - Content and tags are NEVER modified by this operation
        - New metadata fields are added, existing fields are updated
        - updated_at timestamp is automatically updated
        - Use update_memory() if you need to change content or tags
        - Use update_memory_tags() if you only need to change tags
    """
    logger.info(
        f"Tool 'update_memory_metadata' called: memory_id={memory_id}, metadata={metadata}"
    )

    if not memory_id:
        return {"success": False, "error": "memory_id is required"}

    if not metadata:
        return {"success": False, "error": "metadata dictionary cannot be empty"}

    service = MemoryService()

    try:
        result = service.update_memory_metadata(memory_id, metadata)
        return result
    except ValueError as e:
        logger.error(f"❌ update_memory_metadata validation error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"❌ update_memory_metadata failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def merge_memories(
    memory_ids: list[str],
    separator: str = "\n\n---\n\n",
    delete_originals: bool = False,
) -> dict[str, Any]:
    """
    Merge multiple memories into a single new memory.

    Combines content from multiple memories with a separator, merges their tags
    (removing duplicates), and creates a new memory. Optionally deletes originals.
    Useful for consolidating related information or combining conversation threads.

    Args:
        memory_ids: List of memory IDs to merge (minimum 2 required)
        separator: String to separate merged contents (default: "\\n\\n---\\n\\n")
        delete_originals: Whether to delete original memories after merge (default: False)
                          WARNING: Deletion cannot be undone!

    Returns:
        dict: Merge results with new memory details
              Format: {
                  "success": bool,
                  "merged_memory": dict,  # The new combined memory
                  "merged_ids": list[str],  # IDs that were merged
                  "originals_deleted": bool,
                  "merged_count": int
              }

    Examples:
        - merge_memories(["id1", "id2"])
          → Creates new memory with combined content, preserves originals

        - merge_memories(["id1", "id2", "id3"], separator="\\n\\n", delete_originals=True)
          → Merges 3 memories with custom separator and deletes originals

    Notes:
        - Merges tags from all memories (case-insensitive, no duplicates)
        - Uses earliest created_at timestamp from source memories
        - New memory includes metadata: merged_from, merged_count
    """
    logger.info(
        f"Tool 'merge_memories' called: memory_ids={memory_ids}, delete_originals={delete_originals}"
    )

    if not memory_ids:
        return {"success": False, "error": "memory_ids list is required"}

    if len(memory_ids) < 2:
        return {
            "success": False,
            "error": "At least 2 memory IDs are required for merging",
        }

    service = MemoryService()

    try:
        result = service.merge_memories(memory_ids, separator, delete_originals)
        return result
    except Exception as e:
        logger.error(f"❌ merge_memories failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def export_memories(
    format: str = "json",
    filters: dict[str, Any] | None = None,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """
    Export memories to various formats (JSON, Markdown, CSV).

    This tool allows backing up or sharing memories in different formats.
    Supports optional filtering to export specific subsets of memories.

    Args:
        format: Export format - "json", "markdown", or "csv" (default: "json")
                - json: Structured data format, easy to parse
                - markdown: Human-readable format with formatting
                - csv: Spreadsheet-compatible format
        filters: Optional metadata filters to export subset (e.g., {"tags": "python"})
        include_metadata: Include full metadata in export (default: True)

    Returns:
        dict: Export results
              Format: {
                  "success": bool,
                  "format": str,
                  "data": str,  # The formatted export data
                  "count": int,  # Number of memories exported
                  "timestamp": str  # Export timestamp (ISO format)
              }

    Examples:
        - export_memories()
          → Exports all memories to JSON with metadata

        - export_memories(format="markdown", include_metadata=False)
          → Exports to Markdown without metadata

        - export_memories(format="csv", filters={"tags": "python"})
          → Exports Python-tagged memories to CSV

    Notes:
        - JSON format preserves all data types and structure
        - Markdown format is best for human readability
        - CSV format works well with spreadsheet applications
        - Large exports (>1000 memories) may take time
    """
    logger.info(
        f"Tool 'export_memories' called: format={format}, include_metadata={include_metadata}"
    )

    service = MemoryService()

    try:
        result = service.export_memories(format, filters, include_metadata)
        return result
    except Exception as e:
        logger.error(f"❌ export_memories failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def import_memories(
    data: str,
    format: str = "json",
    preserve_ids: bool = False,
    preserve_timestamps: bool = True,
    on_duplicate: str = "skip",
) -> dict[str, Any]:
    """
    Import memories from JSON, Markdown, or CSV format.

    This tool allows restoring from backups or migrating data. Fully compatible
    with export_memories output formats.

    Args:
        data: String data to import (JSON/Markdown/CSV format)
        format: Import format - "json", "markdown", or "csv" (default: "json")
        preserve_ids: Use IDs from import data (default: False - generates new UUIDs)
                      WARNING: True may overwrite existing memories!
        preserve_timestamps: Use timestamps from import (default: True - preserves history)
        on_duplicate: How to handle duplicate content:
                      - "skip": Skip duplicates (default, safest)
                      - "overwrite": Replace existing
                      - "error": Fail on duplicate

    Returns:
        dict: Import results
              Format: {
                  "success": bool,
                  "imported_count": int,
                  "skipped_count": int,
                  "failed_count": int,
                  "errors": list[str],
                  "total_processed": int
              }

    Examples:
        - import_memories(data=json_string)
          → Import from JSON with new IDs, skip duplicates

        - import_memories(data=csv_string, format="csv", preserve_ids=True)
          → Import from CSV preserving original IDs

        - import_memories(data=md_string, format="markdown", on_duplicate="overwrite")
          → Import from Markdown, overwriting duplicates

    Notes:
        - preserve_ids=False (default) is safer - generates new IDs
        - preserve_ids=True can overwrite existing memories with same ID
        - Duplicate detection is based on exact content match
        - Invalid entries are logged but don't stop the import
    """
    logger.info(
        f"Tool 'import_memories' called: format={format}, preserve_ids={preserve_ids}, on_duplicate={on_duplicate}"
    )

    service = MemoryService()

    try:
        result = service.import_memories(
            data=data,
            format=format,
            preserve_ids=preserve_ids,
            preserve_timestamps=preserve_timestamps,
            on_duplicate=on_duplicate,
        )
        return result
    except Exception as e:
        logger.error(f"❌ import_memories failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "imported_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
        }


@mcp.tool()
async def cleanup_duplicates(similarity_threshold: float = 0.95) -> dict[str, Any]:
    """
    Find and remove duplicate memories from the collection.

    This tool identifies memories with identical content and removes duplicates,
    keeping the oldest memory (earliest created_at) from each duplicate group.
    Useful for maintaining data quality and reducing storage usage.

    Args:
        similarity_threshold: Similarity threshold for near-duplicates (0.0-1.0)
                             Default: 0.95 (95% similar)
                             Note: Currently only exact matches are detected

    Returns:
        dict: Cleanup results
              Format: {
                  "success": bool,
                  "duplicates_found": int,
                  "duplicates_removed": int,
                  "exact_matches": int,
                  "similar_matches": int,
                  "kept_memories": list[str],  # IDs of kept memories
                  "removed_memories": list[str]  # IDs of removed memories
              }

    Examples:
        - cleanup_duplicates()
          → Finds and removes all exact duplicate memories

        - cleanup_duplicates(similarity_threshold=0.98)
          → Finds duplicates with 98% similarity (exact matches only for now)

    Notes:
        - Non-destructive: keeps the oldest version of each duplicate
        - Safe to run periodically for maintenance
        - Returns list of removed IDs for audit purposes
        - Future: will support near-duplicate detection with semantic similarity
    """
    logger.info(
        f"Tool 'cleanup_duplicates' called: similarity_threshold={similarity_threshold}"
    )

    service = MemoryService()

    try:
        result = service.cleanup_duplicates(similarity_threshold=similarity_threshold)
        return result
    except Exception as e:
        logger.error(f"❌ cleanup_duplicates failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "duplicates_found": 0,
            "duplicates_removed": 0,
        }


@mcp.tool()
async def optimize_db() -> dict[str, Any]:
    """
    Optimize the database collection for better performance.

    Performs maintenance operations including:
    - Collection statistics analysis
    - Health verification
    - Performance recommendations
    - Automatic index maintenance (handled by ChromaDB)

    Returns:
        dict: Optimization results
              Format: {
                  "success": bool,
                  "collection_name": str,
                  "total_memories": int,
                  "collection_size_estimate": str,
                  "optimizations_performed": list[str],
                  "recommendations": list[str]
              }

    Examples:
        - optimize_db()
          → Analyzes collection and provides optimization recommendations

    Notes:
        - Safe to run anytime - non-destructive operation
        - ChromaDB automatically maintains indices
        - Provides recommendations for improving performance
        - Run periodically (e.g., weekly) for production systems
    """
    logger.info("Tool 'optimize_db' called")

    service = MemoryService()

    try:
        result = service.optimize_db()
        return result
    except Exception as e:
        logger.error(f"❌ optimize_db failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def create_backup() -> dict[str, Any]:
    """
    Create a complete backup of all memories in the collection.

    Exports all memories to JSON format with full metadata and timestamps.
    The backup can be restored later using import_memories() tool.

    Returns:
        dict: Backup results
              Format: {
                  "success": bool,
                  "backup_data": str,  # JSON string with all memories
                  "total_memories": int,
                  "backup_timestamp": str,
                  "collection_name": str
              }

    Examples:
        - create_backup()
          → Creates full backup of all memories in JSON format

    Notes:
        - Backup includes all metadata and timestamps
        - Output is compatible with import_memories() tool
        - Save backup_data to file for long-term storage
        - Consider running before major operations (cleanup, bulk delete)
        - Backup is in JSON format for maximum compatibility

    Typical workflow:
        1. result = create_backup()
        2. Save result["backup_data"] to file: backup_YYYYMMDD.json
        3. Later restore: import_memories(data=backup_data, preserve_ids=True)
    """
    logger.info("Tool 'create_backup' called")

    service = MemoryService()

    try:
        result = service.create_backup()
        return result
    except Exception as e:
        logger.error(f"❌ create_backup failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_embedding(text: str) -> dict[str, Any]:
    """
    Get the embedding vector for a given text.

    Uses the same embedding model as the collection to generate an embedding
    vector for the provided text. Useful for debugging, analysis, or manual
    similarity calculations.

    Args:
        text: Text to generate embedding for

    Returns:
        dict: Embedding information
              Format: {
                  "success": bool,
                  "text": str,
                  "embedding": list[float],  # The embedding vector
                  "dimension": int,  # Vector dimension
                  "model_info": str
              }

    Examples:
        - get_embedding("machine learning")
          → Returns embedding vector with dimension info

        - get_embedding("Python programming")
          → Generates embedding for debugging similarity

    Notes:
        - Same model used for all semantic search operations
        - Embeddings are high-dimensional vectors (typically 384 or 768 dimensions)
        - Can be used to manually calculate similarity between texts
        - Useful for understanding how semantic search works
    """
    logger.info(f"Tool 'get_embedding' called: text='{text[:50]}...'")

    service = MemoryService()

    try:
        result = service.get_embedding(text=text)
        return result
    except Exception as e:
        logger.error(f"❌ get_embedding failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def check_embedding_model() -> dict[str, Any]:
    """
    Get information about the embedding model used by the collection.

    Returns details about the embedding function, model configuration,
    and capabilities. Useful for debugging and understanding how
    semantic search works.

    Returns:
        dict: Model information
              Format: {
                  "success": bool,
                  "model_name": str,
                  "embedding_dimension": int,
                  "collection_name": str,
                  "model_type": str,
                  "details": dict
              }

    Examples:
        - check_embedding_model()
          → Returns model name, dimension, and configuration details

    Notes:
        - ChromaDB uses ONNX-based sentence transformers
        - Typical model: all-MiniLM-L6-v2 (384 dimensions)
        - Supports multilingual text
        - Same model used for all semantic operations
        - Embedding dimension affects memory storage and performance
    """
    logger.info("Tool 'check_embedding_model' called")

    service = MemoryService()

    try:
        result = service.check_embedding_model()
        return result
    except Exception as e:
        logger.error(f"❌ check_embedding_model failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def update_collection_metadata(metadata: str) -> dict[str, Any]:
    """
    Update metadata for the current collection in Chroma Cloud.

    This tool allows you to customize collection metadata with arbitrary key-value pairs.
    Useful for tracking project information, ownership, version, or any custom attributes.

    The metadata is stored in Chroma Cloud and visible in the dashboard.

    Args:
        metadata: JSON or YAML string containing custom metadata key-value pairs.
                 All keys and values must be strings, numbers, or booleans.

                 JSON Example:
                 {
                   "project": "Enigma AI",
                   "owner": "Łukasz",
                   "version": "1.0.0",
                   "environment": "production",
                   "created_date": "2025-11-02"
                 }

                 YAML Example:
                 project: Enigma AI
                 owner: Łukasz
                 version: 1.0.0
                 environment: production
                 created_date: 2025-11-02

    Returns:
        dict: Update results
              Format: {
                  "success": bool,
                  "collection_name": str,
                  "metadata": dict,  # Updated metadata
                  "message": str
              }

    Examples:
        - update_collection_metadata('{"project": "AI Research", "owner": "Łukasz"}')
          → Updates collection metadata with project and owner info

        - update_collection_metadata('''
            project: Enigma
            owner: Łukasz
            version: 2.0.0
          ''')
          → Updates using YAML format

    Notes:
        - Merges with existing metadata (doesn't replace all)
        - To remove a key, set its value to null in JSON or ~ in YAML
        - Metadata is visible in Chroma Cloud dashboard
        - Changes apply immediately to current collection
        - Collection name is determined by URL path (/mcp-{project})
    """
    import json
    import yaml

    logger.info(
        f"Tool 'update_collection_metadata' called with metadata: {metadata[:100]}..."
    )

    try:
        # Try parsing as JSON first
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            # If JSON fails, try YAML
            try:
                metadata_dict = yaml.safe_load(metadata)
            except yaml.YAMLError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON/YAML format: {str(e)}",
                }

        # Validate metadata_dict is a dictionary
        if not isinstance(metadata_dict, dict):
            return {
                "success": False,
                "error": "Metadata must be a JSON/YAML object (dictionary), not a list or scalar value",
            }

        # Get current collection (uses path-based routing context)
        collection = get_collection()

        # Get current metadata
        current_metadata = collection.metadata or {}

        # Merge new metadata with existing
        # Keep system metadata (created_by, description, version)
        updated_metadata = {**current_metadata}

        # Add/update with new metadata
        for key, value in metadata_dict.items():
            # Allow null/None to remove keys
            if value is None:
                updated_metadata.pop(key, None)
            else:
                updated_metadata[key] = value

        # Update collection metadata in Chroma Cloud
        collection.modify(metadata=updated_metadata)

        logger.info(
            f"✅ Collection '{collection.name}' metadata updated: {updated_metadata}"
        )

        return {
            "success": True,
            "collection_name": collection.name,
            "metadata": updated_metadata,
            "message": f"Metadata updated successfully for collection '{collection.name}'",
        }

    except Exception as e:
        logger.error(f"❌ update_collection_metadata failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# RESOURCES & PROMPTS - TODO: Implement later
# ============================================================================
# Resources and prompts will be added in future versions


# ============================================================================
# Server Entry Point
# ============================================================================


def warmup():
    """
    Pre-warm ChromaDB to download ONNX model before server starts.

    This ensures the model is ready and server doesn't respond until fully initialized.
    Updates global warmup state for health check endpoint to track initialization phases.
    """
    global _warmup_complete, _warmup_start_time, _warmup_error

    _warmup_start_time = time.time()
    logger.info("🔄 Warming up ChromaDB and downloading ONNX model...")

    try:
        collection = get_collection()

        # Trigger embedding generation to force model download
        # This downloads ~79MB ONNX model and can take 60-120 seconds
        collection.add(
            ids=["warmup"], documents=["warmup"], metadatas=[{"warmup": True}]
        )

        # Clean up warmup entry
        collection.delete(ids=["warmup"])

        elapsed = time.time() - _warmup_start_time
        logger.info(
            f"✅ ChromaDB warmed up successfully - ONNX model ready ({elapsed:.1f}s)"
        )

        # Mark warmup as complete
        _warmup_complete = True

    except Exception as e:
        logger.error(f"❌ Warmup failed: {e}")
        _warmup_error = str(e)
        raise


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    logger.info("=" * 60)
    logger.info(f"🌳 Yggdrasil v{__version__}")
    logger.info("=" * 60)
    logger.info(f"MCP Protocol: 2025-03-26 (Streamable HTTP)")
    logger.info(
        f"Endpoint: http://{settings.mcp_server_host}:{settings.mcp_server_port}/mcp"
    )
    logger.info(
        f"Healthcheck: http://{settings.mcp_server_host}:{settings.mcp_server_port}/ready"
    )
    logger.info("=" * 60)

    # Skip pre-warmup - let the intelligent readiness probe handle ONNX detection
    # The /ready endpoint will return 503 during ONNX loading and 200 when operational

    # Run with uvicorn
    if os.getenv("RUNNING_IN_PRODUCTION"):
        import multiprocessing

        uvicorn.run(
            "src.server:app",
            host=settings.mcp_server_host,
            port=settings.mcp_server_port,
            workers=(multiprocessing.cpu_count() * 2) + 1,
            timeout_keep_alive=300,  # For SSE connections
        )
    else:
        uvicorn.run(
            "src.server:app",
            host=settings.mcp_server_host,
            port=settings.mcp_server_port,
            reload=False,
            log_level=settings.log_level.lower(),
        )
