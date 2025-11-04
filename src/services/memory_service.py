"""
Memory Service - Core business logic for memory operations
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

from chromadb.api.models.Collection import Collection

from ..config import get_collection

logger = logging.getLogger("yggdrasil.services.memory_service")


class MemoryService:
    """
    Memory service handling all memory operations with Chroma Cloud.

    This service provides core functionality for storing, retrieving,
    searching, and managing memories in vector storage.
    """

    def __init__(self, collection_name: str | None = None):
        """
        Initialize memory service.

        Args:
            collection_name: Optional collection name override
        """
        self.collection: Collection = get_collection(collection_name)
        self.collection_name = self.collection.name
        logger.info(f"MemoryService initialized for collection: {self.collection_name}")

    def store_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a new memory in the collection.

        Args:
            content: Memory content (will be embedded)
            tags: Optional list of tags
            metadata: Optional metadata dictionary

        Returns:
            dict: Created memory information with ID and timestamp
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().timestamp()  # Unix timestamp (float)

        # Build metadata
        memory_metadata = {
            "created_at": timestamp,
            "updated_at": timestamp,
            "tags": ",".join(tags) if tags else "",
        }

        # Add custom metadata
        if metadata:
            memory_metadata.update(metadata)

        try:
            self.collection.add(
                ids=[memory_id],
                documents=[content],
                metadatas=[memory_metadata],
            )

            logger.info(f"✅ Memory stored: {memory_id}")

            return {
                "id": memory_id,
                "content": content,
                "metadata": memory_metadata,
                "timestamp": timestamp,
            }

        except Exception as e:
            logger.error(f"❌ Failed to store memory: {e}")
            raise ValueError(f"Failed to store memory: {e}") from e

    def get_memory(self, memory_id: str) -> dict[str, Any]:
        """
        Retrieve a single memory by ID.

        Args:
            memory_id: The unique identifier of the memory

        Returns:
            dict: Memory with id, content, and metadata

        Raises:
            ValueError: If memory not found
        """
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"]
            )

            if not result["ids"] or len(result["ids"]) == 0:
                logger.warning(f"⚠️ Memory not found: {memory_id}")
                raise ValueError(f"Memory not found: {memory_id}")

            logger.info(f"✅ Memory retrieved: {memory_id}")

            return {
                "id": result["ids"][0],
                "content": result["documents"][0],
                "metadata": result["metadatas"][0]
            }

        except Exception as e:
            logger.error(f"❌ Failed to retrieve memory: {e}")
            raise ValueError(f"Failed to retrieve memory: {e}") from e

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar memories using semantic search.

        Args:
            query: Search query (will be embedded)
            limit: Maximum number of results
            filters: Optional metadata filters (Context7 format)

        Returns:
            list: List of matching memories with scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )

            memories = []
            if results["ids"] and results["ids"][0]:
                for i, memory_id in enumerate(results["ids"][0]):
                    memories.append(
                        {
                            "id": memory_id,
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                        }
                    )

            logger.info(f"🔍 Found {len(memories)} memories for query: '{query[:50]}...'")
            return memories

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise ValueError(f"Search failed: {e}") from e

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            dict: Memory data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"],
            )

            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }

            return None

        except Exception as e:
            logger.error(f"❌ Failed to get memory {memory_id}: {e}")
            return None

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing memory.

        Args:
            memory_id: Memory ID to update
            content: New content (optional)
            tags: New tags (optional)
            metadata: New metadata (optional)

        Returns:
            dict: Updated memory information
        """
        # Get existing memory
        existing = self.get_memory(memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        # Update timestamp
        updated_metadata = existing["metadata"].copy()
        updated_metadata["updated_at"] = datetime.utcnow().timestamp()  # Unix timestamp (float)

        # Update tags if provided
        if tags is not None:
            updated_metadata["tags"] = ",".join(tags)

        # Update custom metadata
        if metadata:
            updated_metadata.update(metadata)

        # Update content if provided
        updated_content = content if content is not None else existing["content"]

        try:
            self.collection.update(
                ids=[memory_id],
                documents=[updated_content],
                metadatas=[updated_metadata],
            )

            logger.info(f"✅ Memory updated: {memory_id}")

            return {
                "id": memory_id,
                "content": updated_content,
                "metadata": updated_metadata,
            }

        except Exception as e:
            logger.error(f"❌ Failed to update memory {memory_id}: {e}")
            raise ValueError(f"Failed to update memory: {e}") from e

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory ID to delete

        Returns:
            bool: True if deleted, False if not found
        """
        try:
            # Check if exists
            if not self.get_memory(memory_id):
                return False

            self.collection.delete(ids=[memory_id])
            logger.info(f"✅ Memory deleted: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete memory {memory_id}: {e}")
            return False

    def list_memories(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        List memories with pagination.

        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            filters: Optional metadata filters

        Returns:
            list: List of memories
        """
        try:
            # Get all IDs first
            all_results = self.collection.get(
                where=filters,
                include=["documents", "metadatas"],
            )

            memories = []
            if all_results["ids"]:
                # Apply pagination
                start = offset
                end = offset + limit

                for i, memory_id in enumerate(all_results["ids"][start:end]):
                    memories.append(
                        {
                            "id": memory_id,
                            "content": all_results["documents"][start + i],
                            "metadata": all_results["metadatas"][start + i],
                        }
                    )

            logger.info(f"📋 Listed {len(memories)} memories")
            return memories

        except Exception as e:
            logger.error(f"❌ Failed to list memories: {e}")
            raise ValueError(f"Failed to list memories: {e}") from e

    def count_memories(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count total memories in collection.

        Args:
            filters: Optional metadata filters

        Returns:
            int: Total count
        """
        try:
            if filters:
                result = self.collection.get(where=filters)
                return len(result["ids"])
            else:
                return self.collection.count()
        except Exception as e:
            logger.error(f"❌ Failed to count memories: {e}")
            return 0

    def search_by_tags(self, tags: list[str]) -> list[dict[str, Any]]:
        """
        Search for memories that contain ANY of the specified tags.

        Uses ANY matching logic - returns memories if they have at least one
        of the specified tags.

        Args:
            tags: List of tags to search for

        Returns:
            list: List of memories matching any of the tags
        """
        if not tags:
            return []

        try:
            # Normalize search tags
            search_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            if not search_tags:
                return []

            # Get all memories
            all_results = self.collection.get(
                include=["documents", "metadatas"]
            )

            memories = []
            if all_results["ids"]:
                for i, memory_id in enumerate(all_results["ids"]):
                    memory_meta = all_results["metadatas"][i]

                    # Parse tags from comma-separated string
                    tags_str = memory_meta.get("tags", "")
                    if tags_str:
                        stored_tags = [tag.strip().lower() for tag in tags_str.split(",") if tag.strip()]

                        # Check if any search tag is in stored tags (ANY logic)
                        if any(search_tag in stored_tags for search_tag in search_tags):
                            memories.append({
                                "id": memory_id,
                                "content": all_results["documents"][i],
                                "metadata": memory_meta,
                            })

            logger.info(f"🏷️  Found {len(memories)} memories matching tags: {tags}")
            return memories

        except Exception as e:
            logger.error(f"❌ Failed to search by tags: {e}")
            raise ValueError(f"Failed to search by tags: {e}") from e

    def search_by_all_tags(self, tags: list[str]) -> list[dict[str, Any]]:
        """
        Search for memories by tags using ALL matching logic.

        Returns memories that contain ALL of the specified tags. Perfect for
        finding memories with specific combinations of tags.

        Args:
            tags: List of tags to search for. Returns memories containing ALL of these tags.

        Returns:
            list: List of memories matching all tags
        """
        if not tags:
            return []

        try:
            # Normalize search tags
            search_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            if not search_tags:
                return []

            search_tags_set = set(search_tags)

            # Get all memories
            all_results = self.collection.get(
                include=["documents", "metadatas"]
            )

            memories = []
            if all_results["ids"]:
                for i, memory_id in enumerate(all_results["ids"]):
                    memory_meta = all_results["metadatas"][i]

                    # Parse tags from comma-separated string
                    tags_str = memory_meta.get("tags", "")
                    if tags_str:
                        stored_tags = [tag.strip().lower() for tag in tags_str.split(",") if tag.strip()]
                        stored_tags_set = set(stored_tags)

                        # Check if ALL search tags are in stored tags (ALL logic)
                        if search_tags_set.issubset(stored_tags_set):
                            memories.append({
                                "id": memory_id,
                                "content": all_results["documents"][i],
                                "metadata": memory_meta,
                            })

            logger.info(f"🏷️  Found {len(memories)} memories with ALL tags: {tags}")
            return memories

        except Exception as e:
            logger.error(f"❌ Failed to search by all tags: {e}")
            raise ValueError(f"Failed to search by all tags: {e}") from e

    def debug_retrieve(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = -1.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories with debug information for semantic search analysis.

        Returns memories with raw similarity scores, distances, and memory IDs
        to help understand search quality and behavior.

        Note: Similarity scores can be negative when distance > 1.0. Use a low
        threshold (like -1.0) to capture all results for debugging.

        Args:
            query: Search query for debugging retrieval
            limit: Maximum number of results (default: 10)
            similarity_threshold: Minimum similarity score, can be negative (default: -1.0)

        Returns:
            list: Memories with debug info (raw_similarity, raw_distance, memory_id)
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            memories = []
            if results["ids"] and results["ids"][0]:
                for i, memory_id in enumerate(results["ids"][0]):
                    raw_distance = results["distances"][0][i]
                    raw_similarity = 1 - raw_distance

                    # Apply similarity threshold filter
                    if raw_similarity >= similarity_threshold:
                        memories.append(
                            {
                                "id": memory_id,
                                "content": results["documents"][0][i],
                                "metadata": results["metadatas"][0][i],
                                "score": round(raw_similarity, 4),
                                "debug_info": {
                                    "raw_similarity": round(raw_similarity, 4),
                                    "raw_distance": round(raw_distance, 4),
                                    "memory_id": memory_id,
                                },
                            }
                        )

            logger.info(
                f"🐛 Debug retrieve found {len(memories)} memories (threshold: {similarity_threshold}) for: '{query[:50]}...'"
            )
            return memories

        except Exception as e:
            logger.error(f"❌ Debug retrieve failed: {e}")
            raise ValueError(f"Debug retrieve failed: {e}") from e

    def delete_by_tags(self, tags: list[str]) -> dict[str, Any]:
        """
        Delete all memories containing ANY of the specified tags.

        WARNING: This permanently deletes memories and cannot be undone!
        Uses ANY matching logic - deletes memories with at least one of the tags.

        Args:
            tags: List of tags. Memories containing ANY of these tags will be deleted.

        Returns:
            dict: Deletion results with count and matched tags
        """
        if not tags:
            return {"success": False, "deleted_count": 0, "error": "Tags are required"}

        try:
            # Normalize search tags
            search_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            if not search_tags:
                return {"success": False, "deleted_count": 0, "error": "No valid tags provided"}

            # Get all memories
            all_results = self.collection.get(
                include=["metadatas"]
            )

            ids_to_delete = []
            matched_tags = set()

            if all_results["ids"]:
                for i, memory_id in enumerate(all_results["ids"]):
                    memory_meta = all_results["metadatas"][i]

                    # Parse tags from comma-separated string
                    tags_str = memory_meta.get("tags", "")
                    if tags_str:
                        stored_tags = [tag.strip().lower() for tag in tags_str.split(",") if tag.strip()]

                        # Check if any search tag is in stored tags (ANY logic)
                        for search_tag in search_tags:
                            if search_tag in stored_tags:
                                ids_to_delete.append(memory_id)
                                matched_tags.add(search_tag)
                                break  # No need to check other tags for this memory

            if not ids_to_delete:
                logger.info(f"🏷️  No memories found with tags: {tags}")
                return {
                    "success": True,
                    "deleted_count": 0,
                    "message": f"No memories found with tags: {', '.join(tags)}"
                }

            # Delete the memories
            self.collection.delete(ids=ids_to_delete)
            logger.warning(f"⚠️  Deleted {len(ids_to_delete)} memories with tags: {list(matched_tags)}")

            return {
                "success": True,
                "deleted_count": len(ids_to_delete),
                "matched_tags": list(matched_tags),
                "message": f"Deleted {len(ids_to_delete)} memories"
            }

        except Exception as e:
            logger.error(f"❌ Failed to delete by tags: {e}")
            raise ValueError(f"Failed to delete by tags: {e}") from e

    def delete_by_all_tags(self, tags: list[str]) -> dict[str, Any]:
        """
        Delete memories that contain ALL of the specified tags.

        WARNING: This permanently deletes memories and cannot be undone!
        Uses ALL matching logic - deletes memories only if they contain EVERY specified tag.

        Args:
            tags: List of tags. Memories containing ALL of these tags will be deleted.

        Returns:
            dict: Deletion results with count and matched tags
        """
        if not tags:
            return {"success": False, "deleted_count": 0, "error": "Tags are required"}

        try:
            # Normalize search tags
            search_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            if not search_tags:
                return {"success": False, "deleted_count": 0, "error": "No valid tags provided"}

            search_tags_set = set(search_tags)

            # Get all memories
            all_results = self.collection.get(
                include=["metadatas"]
            )

            ids_to_delete = []

            if all_results["ids"]:
                for i, memory_id in enumerate(all_results["ids"]):
                    memory_meta = all_results["metadatas"][i]

                    # Parse tags from comma-separated string
                    tags_str = memory_meta.get("tags", "")
                    if tags_str:
                        stored_tags = [tag.strip().lower() for tag in tags_str.split(",") if tag.strip()]
                        stored_tags_set = set(stored_tags)

                        # Check if ALL search tags are in stored tags (ALL logic)
                        if search_tags_set.issubset(stored_tags_set):
                            ids_to_delete.append(memory_id)

            if not ids_to_delete:
                logger.info(f"🏷️  No memories found with ALL tags: {tags}")
                return {
                    "success": True,
                    "deleted_count": 0,
                    "message": f"No memories found with ALL tags: {', '.join(tags)}"
                }

            # Delete the memories
            self.collection.delete(ids=ids_to_delete)
            logger.warning(f"⚠️  Deleted {len(ids_to_delete)} memories with ALL tags: {search_tags}")

            return {
                "success": True,
                "deleted_count": len(ids_to_delete),
                "matched_tags": search_tags,
                "message": f"Deleted {len(ids_to_delete)} memories with ALL tags"
            }

        except Exception as e:
            logger.error(f"❌ Failed to delete by all tags: {e}")
            raise ValueError(f"Failed to delete by all tags: {e}") from e

    def update_memory_tags(
        self,
        memory_id: str,
        tags: list[str],
        mode: str = "replace"
    ) -> dict[str, Any]:
        """
        Update tags for an existing memory without modifying content.

        This is a specialized method for tag updates - more efficient and safer
        than using update_memory() when you only want to change tags.

        Args:
            memory_id: Memory ID to update
            tags: List of tags to apply
            mode: Update mode - "replace" (default), "append", or "remove"
                  - replace: Replace all existing tags with new ones
                  - append: Add new tags to existing tags (no duplicates)
                  - remove: Remove specified tags from existing tags

        Returns:
            dict: Updated memory info with success status and tag changes

        Raises:
            ValueError: If memory not found or invalid mode
        """
        # Validate mode
        valid_modes = ["replace", "append", "remove"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Get existing memory
        existing = self.get_memory(memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        # Get current tags
        current_tags_str = existing["metadata"].get("tags", "")
        current_tags = [tag.strip() for tag in current_tags_str.split(",") if tag.strip()]

        # Normalize input tags
        new_tags = [tag.strip().lower() for tag in tags if tag.strip()]

        # Apply mode
        if mode == "replace":
            final_tags = new_tags
            operation = "replaced"
        elif mode == "append":
            # Add new tags, avoid duplicates (case-insensitive)
            current_tags_lower = [tag.lower() for tag in current_tags]
            for tag in new_tags:
                if tag not in current_tags_lower:
                    current_tags.append(tag)
            final_tags = current_tags
            operation = "appended"
        elif mode == "remove":
            # Remove specified tags (case-insensitive)
            new_tags_set = set(new_tags)
            final_tags = [tag for tag in current_tags if tag.lower() not in new_tags_set]
            operation = "removed"

        # Update metadata
        updated_metadata = existing["metadata"].copy()
        updated_metadata["updated_at"] = datetime.utcnow().timestamp()
        updated_metadata["tags"] = ",".join(final_tags) if final_tags else ""

        try:
            # Update only metadata, keep content unchanged
            self.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )

            logger.info(f"✅ Tags {operation} for memory {memory_id}: {final_tags}")

            return {
                "success": True,
                "id": memory_id,
                "operation": operation,
                "previous_tags": current_tags,
                "new_tags": final_tags,
                "metadata": updated_metadata
            }

        except Exception as e:
            logger.error(f"❌ Failed to update tags for memory {memory_id}: {e}")
            raise ValueError(f"Failed to update tags: {e}") from e

    def merge_memories(
        self,
        memory_ids: list[str],
        separator: str = "\n\n---\n\n",
        delete_originals: bool = False
    ) -> dict[str, Any]:
        """
        Merge multiple memories into a single new memory.

        This combines the content of multiple memories with a separator,
        merges their tags (removing duplicates), and creates a new memory
        with combined metadata. Optionally deletes the original memories.

        Args:
            memory_ids: List of memory IDs to merge (minimum 2 required)
            separator: String to use between merged contents (default: "\n\n---\n\n")
            delete_originals: Whether to delete original memories after merge (default: False)

        Returns:
            dict: Merge results with new memory info
                  Format: {
                      "success": bool,
                      "merged_memory": dict,
                      "merged_ids": list[str],
                      "originals_deleted": bool,
                      "merged_count": int
                  }

        Raises:
            ValueError: If less than 2 IDs provided or any memory not found
        """
        # Validate input
        if not memory_ids or len(memory_ids) < 2:
            raise ValueError("At least 2 memory IDs are required for merging")

        try:
            # Fetch all memories
            memories = []
            missing_ids = []

            for memory_id in memory_ids:
                memory = self.get_memory(memory_id)
                if not memory:
                    missing_ids.append(memory_id)
                else:
                    memories.append(memory)

            if missing_ids:
                raise ValueError(f"Memories not found: {missing_ids}")

            # Merge content
            contents = [mem["content"] for mem in memories]
            merged_content = separator.join(contents)

            # Merge tags (union, case-insensitive, no duplicates)
            all_tags = []
            for mem in memories:
                tags_str = mem["metadata"].get("tags", "")
                if tags_str:
                    tags = [tag.strip().lower() for tag in tags_str.split(",") if tag.strip()]
                    all_tags.extend(tags)

            # Remove duplicates while preserving order
            unique_tags = []
            seen = set()
            for tag in all_tags:
                if tag not in seen:
                    unique_tags.append(tag)
                    seen.add(tag)

            # Get earliest created_at and latest updated_at
            created_ats = [mem["metadata"].get("created_at", 0) for mem in memories]
            earliest_created = min(created_ats) if created_ats else datetime.utcnow().timestamp()

            # Create merged memory metadata
            merged_metadata = {
                "created_at": earliest_created,
                "updated_at": datetime.utcnow().timestamp(),
                "tags": ",".join(unique_tags) if unique_tags else "",
                "merged_from": ",".join(memory_ids),
                "merged_count": len(memory_ids)
            }

            # Store new merged memory
            new_memory = self.store_memory(
                content=merged_content,
                tags=unique_tags,
                metadata={"merged_from": merged_metadata["merged_from"], "merged_count": merged_metadata["merged_count"]}
            )

            # Delete originals if requested
            originals_deleted = False
            if delete_originals:
                for memory_id in memory_ids:
                    self.delete_memory(memory_id)
                originals_deleted = True
                logger.info(f"🔗 Merged {len(memory_ids)} memories and deleted originals")
            else:
                logger.info(f"🔗 Merged {len(memory_ids)} memories (originals preserved)")

            return {
                "success": True,
                "merged_memory": new_memory,
                "merged_ids": memory_ids,
                "originals_deleted": originals_deleted,
                "merged_count": len(memory_ids)
            }

        except Exception as e:
            logger.error(f"❌ Failed to merge memories: {e}")
            raise ValueError(f"Failed to merge memories: {e}") from e

    def export_memories(
        self,
        format: str = "json",
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True
    ) -> dict[str, Any]:
        """
        Export memories to various formats (JSON, Markdown, CSV).

        This allows backing up memories or sharing them in different formats.
        Supports optional filtering before export.

        Args:
            format: Export format - "json", "markdown", or "csv" (default: "json")
            filters: Optional metadata filters to export subset of memories
            include_metadata: Whether to include full metadata (default: True)

        Returns:
            dict: Export results with data string
                  Format: {
                      "success": bool,
                      "format": str,
                      "data": str,  # Formatted export data
                      "count": int,  # Number of memories exported
                      "timestamp": str  # Export timestamp
                  }

        Raises:
            ValueError: If invalid format specified
        """
        # Validate format
        valid_formats = ["json", "markdown", "csv"]
        format = format.lower()
        if format not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Must be one of: {valid_formats}")

        try:
            # Get memories (with optional filters)
            memories = self.list_memories(limit=10000, filters=filters)

            if not memories:
                return {
                    "success": True,
                    "format": format,
                    "data": "",
                    "count": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "No memories to export"
                }

            export_timestamp = datetime.utcnow().isoformat()

            # Export based on format
            if format == "json":
                data = self._export_to_json(memories, include_metadata)
            elif format == "markdown":
                data = self._export_to_markdown(memories, include_metadata)
            elif format == "csv":
                data = self._export_to_csv(memories, include_metadata)

            logger.info(f"📤 Exported {len(memories)} memories to {format.upper()}")

            return {
                "success": True,
                "format": format,
                "data": data,
                "count": len(memories),
                "timestamp": export_timestamp
            }

        except Exception as e:
            logger.error(f"❌ Failed to export memories: {e}")
            raise ValueError(f"Failed to export memories: {e}") from e

    def _export_to_json(self, memories: list[dict[str, Any]], include_metadata: bool) -> str:
        """Export memories to JSON format."""
        export_data = []
        for mem in memories:
            entry = {
                "id": mem["id"],
                "content": mem["content"]
            }
            if include_metadata:
                entry["metadata"] = mem["metadata"]
            export_data.append(entry)

        return json.dumps(export_data, indent=2, ensure_ascii=False)

    def _export_to_markdown(self, memories: list[dict[str, Any]], include_metadata: bool) -> str:
        """Export memories to Markdown format."""
        lines = ["# Memory Export", ""]

        for i, mem in enumerate(memories, 1):
            lines.append(f"## Memory {i}")
            lines.append("")

            if include_metadata:
                metadata = mem["metadata"]
                lines.append(f"**ID:** `{mem['id']}`")

                # Format timestamps
                created_at = metadata.get("created_at")
                if isinstance(created_at, (int, float)):
                    created_str = datetime.fromtimestamp(created_at).isoformat()
                else:
                    created_str = str(created_at)
                lines.append(f"**Created:** {created_str}")

                # Tags
                tags = metadata.get("tags", "")
                if tags:
                    tags_list = [f"`{t.strip()}`" for t in tags.split(",") if t.strip()]
                    lines.append(f"**Tags:** {', '.join(tags_list)}")

                lines.append("")

            lines.append(mem["content"])
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _export_to_csv(self, memories: list[dict[str, Any]], include_metadata: bool) -> str:
        """Export memories to CSV format."""
        lines = []

        # Header
        if include_metadata:
            lines.append("id,content,tags,created_at,updated_at")
        else:
            lines.append("id,content")

        # Rows
        for mem in memories:
            # Escape quotes in content
            content = mem["content"].replace('"', '""')

            if include_metadata:
                metadata = mem["metadata"]
                tags = metadata.get("tags", "")
                created = metadata.get("created_at", "")
                updated = metadata.get("updated_at", "")

                lines.append(f'"{mem["id"]}","{content}","{tags}","{created}","{updated}"')
            else:
                lines.append(f'"{mem["id"]}","{content}"')

        return "\n".join(lines)

    def import_memories(
        self,
        data: str,
        format: str = "json",
        preserve_ids: bool = False,
        preserve_timestamps: bool = True,
        on_duplicate: str = "skip"
    ) -> dict[str, Any]:
        """
        Import memories from JSON, Markdown, or CSV format.

        This allows restoring from backups or migrating data. Compatible with
        export_memories formats.

        Args:
            data: String data to import (JSON/Markdown/CSV)
            format: Import format - "json", "markdown", or "csv" (default: "json")
            preserve_ids: Use IDs from import data (default: False - generates new IDs)
            preserve_timestamps: Use timestamps from import (default: True)
            on_duplicate: How to handle duplicates - "skip", "overwrite", "error" (default: "skip")

        Returns:
            dict: Import results
                  Format: {
                      "success": bool,
                      "imported_count": int,
                      "skipped_count": int,
                      "failed_count": int,
                      "errors": list[str]
                  }

        Raises:
            ValueError: If invalid format or on_duplicate value
        """
        # Validate format
        valid_formats = ["json", "markdown", "csv"]
        format = format.lower()
        if format not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Must be one of: {valid_formats}")

        # Validate on_duplicate
        valid_duplicate_handlers = ["skip", "overwrite", "error"]
        if on_duplicate not in valid_duplicate_handlers:
            raise ValueError(f"Invalid on_duplicate '{on_duplicate}'. Must be one of: {valid_duplicate_handlers}")

        try:
            # Parse based on format
            if format == "json":
                memories = self._import_from_json(data)
            elif format == "markdown":
                memories = self._import_from_markdown(data)
            elif format == "csv":
                memories = self._import_from_csv(data)

            # Import memories
            imported_count = 0
            skipped_count = 0
            failed_count = 0
            errors = []

            for mem_data in memories:
                try:
                    content = mem_data.get("content")
                    if not content:
                        failed_count += 1
                        errors.append(f"Missing content in entry: {mem_data.get('id', 'unknown')}")
                        continue

                    # Check for duplicates if on_duplicate != "overwrite"
                    if on_duplicate != "overwrite":
                        # Duplicate check: get all documents and search for exact content match
                        # Note: ChromaDB where clause only works on metadata, not document content
                        all_docs = self.collection.get(
                            include=["documents"]
                        )
                        is_duplicate = False
                        if all_docs["documents"]:
                            for doc in all_docs["documents"]:
                                if doc == content:
                                    is_duplicate = True
                                    break

                        if is_duplicate:
                            if on_duplicate == "error":
                                raise ValueError(f"Duplicate content found: {content[:50]}...")
                            elif on_duplicate == "skip":
                                skipped_count += 1
                                continue

                    # Prepare metadata
                    metadata = mem_data.get("metadata", {})
                    tags_str = metadata.get("tags", "")
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []

                    # Handle timestamps
                    if preserve_timestamps:
                        created_at = metadata.get("created_at")
                        updated_at = metadata.get("updated_at")

                        # Convert ISO strings to Unix floats if needed
                        if created_at:
                            if isinstance(created_at, str):
                                try:
                                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                                    created_at = dt.timestamp()
                                except ValueError:
                                    created_at = None
                            elif not isinstance(created_at, (int, float)):
                                created_at = None

                        if updated_at:
                            if isinstance(updated_at, str):
                                try:
                                    dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                                    updated_at = dt.timestamp()
                                except ValueError:
                                    updated_at = None
                            elif not isinstance(updated_at, (int, float)):
                                updated_at = None
                    else:
                        created_at = None
                        updated_at = None

                    # Store memory
                    if preserve_ids and mem_data.get("id"):
                        # Use provided ID (risky - might overwrite!)
                        memory_id = mem_data["id"]
                        final_metadata = {
                            "created_at": created_at or datetime.utcnow().timestamp(),
                            "updated_at": updated_at or datetime.utcnow().timestamp(),
                            "tags": ",".join(tags) if tags else ""
                        }
                        # Add any extra metadata
                        for key, value in metadata.items():
                            if key not in ["created_at", "updated_at", "tags"]:
                                final_metadata[key] = value

                        self.collection.add(
                            ids=[memory_id],
                            documents=[content],
                            metadatas=[final_metadata]
                        )
                    else:
                        # Generate new ID via store_memory
                        custom_metadata = {}
                        if preserve_timestamps:
                            if created_at is not None:
                                custom_metadata["created_at"] = created_at
                            if updated_at is not None:
                                custom_metadata["updated_at"] = updated_at

                        self.store_memory(
                            content=content,
                            tags=tags,
                            metadata=custom_metadata if custom_metadata else None
                        )

                    imported_count += 1

                except Exception as e:
                    failed_count += 1
                    errors.append(f"Failed to import entry: {str(e)}")

            logger.info(f"📥 Imported {imported_count} memories ({skipped_count} skipped, {failed_count} failed)")

            return {
                "success": True,
                "imported_count": imported_count,
                "skipped_count": skipped_count,
                "failed_count": failed_count,
                "errors": errors,
                "total_processed": len(memories)
            }

        except Exception as e:
            logger.error(f"❌ Failed to import memories: {e}")
            raise ValueError(f"Failed to import memories: {e}") from e

    def _import_from_json(self, data: str) -> list[dict[str, Any]]:
        """Parse JSON import data."""
        try:
            parsed = json.loads(data)
            if not isinstance(parsed, list):
                raise ValueError("JSON data must be an array of memory objects")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

    def _import_from_markdown(self, data: str) -> list[dict[str, Any]]:
        """Parse Markdown import data."""
        memories = []
        lines = data.split("\n")

        current_memory = {}
        in_content = False
        content_lines = []

        for line in lines:
            if line.startswith("## Memory"):
                # Save previous memory
                if current_memory and content_lines:
                    current_memory["content"] = "\n".join(content_lines).strip()
                    memories.append(current_memory)

                # Start new memory
                current_memory = {"metadata": {}}
                content_lines = []
                in_content = False

            elif line.startswith("**ID:**"):
                memory_id = line.replace("**ID:**", "").strip().strip("`")
                current_memory["id"] = memory_id

            elif line.startswith("**Tags:**"):
                tags_part = line.replace("**Tags:**", "").strip()
                # Remove backticks and join
                tags = [t.strip().strip("`") for t in tags_part.split(",")]
                current_memory["metadata"]["tags"] = ",".join(tags)

            elif line.startswith("**Created:**"):
                # Skip timestamp parsing for now (optional)
                pass

            elif line == "---":
                in_content = False

            elif line and not line.startswith("#") and not line.startswith("**"):
                if in_content or current_memory.get("id"):
                    in_content = True
                    content_lines.append(line)

        # Save last memory
        if current_memory and content_lines:
            current_memory["content"] = "\n".join(content_lines).strip()
            memories.append(current_memory)

        return memories

    def _import_from_csv(self, data: str) -> list[dict[str, Any]]:
        """Parse CSV import data."""
        memories = []
        lines = data.strip().split("\n")

        if not lines:
            return memories

        # Parse header
        header = lines[0].split(",")

        # Parse rows
        for line in lines[1:]:
            if not line.strip():
                continue

            # Simple CSV parsing (handles quoted fields)
            parts = []
            current = []
            in_quotes = False

            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == "," and not in_quotes:
                    parts.append("".join(current).strip('"'))
                    current = []
                else:
                    current.append(char)
            parts.append("".join(current).strip('"'))

            if len(parts) >= 2:
                memory = {
                    "id": parts[0],
                    "content": parts[1].replace('""', '"'),  # Unescape quotes
                    "metadata": {}
                }

                # Parse metadata if present
                if len(parts) >= 3:
                    memory["metadata"]["tags"] = parts[2]
                if len(parts) >= 4:
                    try:
                        memory["metadata"]["created_at"] = float(parts[3])
                    except ValueError:
                        pass
                if len(parts) >= 5:
                    try:
                        memory["metadata"]["updated_at"] = float(parts[4])
                    except ValueError:
                        pass

                memories.append(memory)

        return memories

    def check_health(self) -> dict[str, Any]:
        """
        Perform health check on the memory system.

        Checks:
            - ChromaDB connection status
            - Collection accessibility
            - Memory count
            - Basic query functionality

        Returns:
            dict: Health status with details
                {
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
                    "errors": list[str]
                }
        """
        health_status = {
            "status": "healthy",
            "checks": {
                "chromadb_connection": False,
                "collection_accessible": False,
                "query_functional": False
            },
            "stats": {
                "total_memories": 0,
                "collection_name": self.collection_name
            },
            "errors": []
        }

        try:
            # Check 1: Collection accessible
            try:
                result = self.collection.get(limit=1)
                health_status["checks"]["collection_accessible"] = True
                health_status["checks"]["chromadb_connection"] = True
                logger.info("✅ ChromaDB connection: OK")
            except Exception as e:
                health_status["errors"].append(f"Collection access failed: {str(e)}")
                health_status["status"] = "unhealthy"
                logger.error(f"❌ Collection access failed: {e}")
                return health_status

            # Check 2: Get memory count
            try:
                count_result = self.collection.count()
                health_status["stats"]["total_memories"] = count_result
                logger.info(f"✅ Memory count: {count_result}")
            except Exception as e:
                health_status["errors"].append(f"Count operation failed: {str(e)}")
                health_status["status"] = "degraded"
                logger.warning(f"⚠️ Count operation failed: {e}")

            # Check 3: Test query functionality (if we have memories)
            if health_status["stats"]["total_memories"] > 0:
                try:
                    test_result = self.collection.query(
                        query_texts=["health check test"],
                        n_results=1
                    )
                    if test_result and "ids" in test_result:
                        health_status["checks"]["query_functional"] = True
                        logger.info("✅ Query functionality: OK")
                except Exception as e:
                    health_status["errors"].append(f"Query test failed: {str(e)}")
                    health_status["status"] = "degraded"
                    logger.warning(f"⚠️ Query test failed: {e}")
            else:
                # No memories to test with, but that's okay
                health_status["checks"]["query_functional"] = True
                logger.info("✅ Query functionality: OK (no memories to test)")

            # Final status determination
            if all(health_status["checks"].values()):
                health_status["status"] = "healthy"
                logger.info("✅ Health check: HEALTHY")
            elif health_status["checks"]["chromadb_connection"]:
                health_status["status"] = "degraded"
                logger.warning("⚠️ Health check: DEGRADED")
            else:
                health_status["status"] = "unhealthy"
                logger.error("❌ Health check: UNHEALTHY")

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(f"Health check failed: {str(e)}")
            logger.error(f"❌ Health check failed: {e}")

        return health_status

    def delete_by_timeframe(
        self,
        start_time: float,
        end_time: float
    ) -> dict[str, Any]:
        """
        Delete memories within a specific timeframe.

        Deletes all memories where created_at is between start_time and end_time (inclusive).

        Args:
            start_time: Start timestamp (Unix timestamp as float)
            end_time: End timestamp (Unix timestamp as float)

        Returns:
            dict: Deletion results
                {
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
        """
        if start_time > end_time:
            raise ValueError("start_time must be less than or equal to end_time")

        try:
            # Build where clause for timeframe filtering
            where_clause = {
                "$and": [
                    {"created_at": {"$gte": start_time}},
                    {"created_at": {"$lte": end_time}}
                ]
            }

            # Get memories within timeframe
            result = self.collection.get(
                where=where_clause,
                include=["metadatas"]
            )

            deleted_count = len(result["ids"]) if result["ids"] else 0

            if deleted_count > 0:
                # Delete memories
                self.collection.delete(ids=result["ids"])
                logger.warning(
                    f"⚠️  Deleted {deleted_count} memories from timeframe "
                    f"{datetime.fromtimestamp(start_time).isoformat()} to "
                    f"{datetime.fromtimestamp(end_time).isoformat()}"
                )
            else:
                logger.info("No memories found in specified timeframe")

            return {
                "success": True,
                "deleted_count": deleted_count,
                "timeframe": {
                    "start": start_time,
                    "end": end_time,
                    "start_iso": datetime.fromtimestamp(start_time).isoformat(),
                    "end_iso": datetime.fromtimestamp(end_time).isoformat()
                },
                "message": f"Deleted {deleted_count} memories from timeframe"
            }

        except Exception as e:
            logger.error(f"❌ Failed to delete memories by timeframe: {e}")
            raise ValueError(f"Failed to delete memories by timeframe: {e}") from e

    def delete_before_date(self, timestamp: float) -> dict[str, Any]:
        """
        Delete all memories created before a specific date.

        Deletes all memories where created_at < timestamp.

        Args:
            timestamp: Cutoff timestamp (Unix timestamp as float)
                      All memories created before this time will be deleted

        Returns:
            dict: Deletion results
                {
                    "success": bool,
                    "deleted_count": int,
                    "cutoff_date": {
                        "timestamp": float,
                        "iso": str
                    },
                    "message": str
                }
        """
        try:
            # Build where clause for date filtering
            where_clause = {
                "created_at": {"$lt": timestamp}
            }

            # Get memories before date
            result = self.collection.get(
                where=where_clause,
                include=["metadatas"]
            )

            deleted_count = len(result["ids"]) if result["ids"] else 0

            if deleted_count > 0:
                # Delete memories
                self.collection.delete(ids=result["ids"])
                logger.warning(
                    f"⚠️  Deleted {deleted_count} memories created before "
                    f"{datetime.fromtimestamp(timestamp).isoformat()}"
                )
            else:
                logger.info(f"No memories found before {datetime.fromtimestamp(timestamp).isoformat()}")

            return {
                "success": True,
                "deleted_count": deleted_count,
                "cutoff_date": {
                    "timestamp": timestamp,
                    "iso": datetime.fromtimestamp(timestamp).isoformat()
                },
                "message": f"Deleted {deleted_count} memories created before {datetime.fromtimestamp(timestamp).isoformat()}"
            }

        except Exception as e:
            logger.error(f"❌ Failed to delete memories before date: {e}")
            raise ValueError(f"Failed to delete memories before date: {e}") from e

    def cleanup_duplicates(self, similarity_threshold: float = 0.95) -> dict[str, Any]:
        """
        Find and remove duplicate memories.

        Finds memories with identical or very similar content and removes duplicates,
        keeping the oldest memory (earliest created_at) from each duplicate group.

        Args:
            similarity_threshold: Similarity threshold for near-duplicates (0.0-1.0)
                                 Default: 0.95 (95% similar)

        Returns:
            dict: Cleanup results
                {
                    "success": bool,
                    "duplicates_found": int,
                    "duplicates_removed": int,
                    "exact_matches": int,
                    "similar_matches": int,
                    "kept_memories": list[str],  # IDs of kept memories
                    "removed_memories": list[str]  # IDs of removed memories
                }
        """
        try:
            # Get all memories
            all_results = self.collection.get(
                include=["documents", "metadatas"]
            )

            if not all_results["ids"]:
                return {
                    "success": True,
                    "duplicates_found": 0,
                    "duplicates_removed": 0,
                    "exact_matches": 0,
                    "similar_matches": 0,
                    "kept_memories": [],
                    "removed_memories": []
                }

            # Group memories by content for exact duplicates
            content_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}

            for i, memory_id in enumerate(all_results["ids"]):
                content = all_results["documents"][i]
                metadata = all_results["metadatas"][i]

                if content not in content_groups:
                    content_groups[content] = []
                content_groups[content].append((memory_id, metadata))

            # Find exact duplicates
            exact_duplicates = []
            kept_memories = []
            removed_memories = []

            for content, memories in content_groups.items():
                if len(memories) > 1:
                    # Sort by created_at, keep oldest
                    sorted_mems = sorted(
                        memories,
                        key=lambda x: x[1].get("created_at", float('inf'))
                    )
                    kept_memories.append(sorted_mems[0][0])
                    for mem_id, _ in sorted_mems[1:]:
                        exact_duplicates.append(mem_id)
                        removed_memories.append(mem_id)

            # Delete exact duplicates
            if exact_duplicates:
                self.collection.delete(ids=exact_duplicates)
                logger.warning(
                    f"⚠️  Removed {len(exact_duplicates)} exact duplicate memories, "
                    f"kept {len(kept_memories)} oldest versions"
                )

            return {
                "success": True,
                "duplicates_found": len(exact_duplicates),
                "duplicates_removed": len(exact_duplicates),
                "exact_matches": len(exact_duplicates),
                "similar_matches": 0,  # TODO: implement near-duplicate detection
                "kept_memories": kept_memories,
                "removed_memories": removed_memories
            }

        except Exception as e:
            logger.error(f"❌ Failed to cleanup duplicates: {e}")
            raise ValueError(f"Failed to cleanup duplicates: {e}") from e

    def optimize_db(self) -> dict[str, Any]:
        """
        Optimize the database collection.

        Performs maintenance operations to improve performance:
        - Collects statistics about the collection
        - Verifies collection health
        - Reports on optimization opportunities

        Returns:
            dict: Optimization results
                {
                    "success": bool,
                    "collection_name": str,
                    "total_memories": int,
                    "collection_size_estimate": str,
                    "optimizations_performed": list[str],
                    "recommendations": list[str]
                }
        """
        try:
            # Get collection stats
            count = self.collection.count()

            # Get sample to estimate size
            sample = self.collection.get(limit=min(100, count), include=["documents", "metadatas"])

            optimizations = []
            recommendations = []

            # Analyze collection
            if sample["documents"]:
                avg_content_length = sum(len(doc) for doc in sample["documents"]) / len(sample["documents"])
                optimizations.append(f"Analyzed {len(sample['documents'])} sample memories")

                if avg_content_length > 5000:
                    recommendations.append("Consider splitting very long memories for better search performance")

                # Check for missing metadata
                missing_created_at = sum(
                    1 for meta in sample["metadatas"]
                    if not meta.get("created_at")
                )
                if missing_created_at > 0:
                    recommendations.append(f"Found {missing_created_at} memories without created_at timestamp")

            # Collection is optimized automatically by ChromaDB
            optimizations.append("Collection indices are maintained automatically by ChromaDB")

            logger.info(f"✅ Database optimization completed: {count} memories analyzed")

            return {
                "success": True,
                "collection_name": self.collection_name,
                "total_memories": count,
                "collection_size_estimate": f"~{count * 500 / 1024:.2f} KB",  # Rough estimate
                "optimizations_performed": optimizations,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.error(f"❌ Failed to optimize database: {e}")
            raise ValueError(f"Failed to optimize database: {e}") from e

    def create_backup(self) -> dict[str, Any]:
        """
        Create a backup of all memories in the collection.

        Exports all memories to JSON format with full metadata and timestamps.
        The backup can be restored using import_memories().

        Returns:
            dict: Backup results
                {
                    "success": bool,
                    "backup_data": str,  # JSON string with all memories
                    "total_memories": int,
                    "backup_timestamp": str,
                    "collection_name": str
                }
        """
        try:
            # Get all memories
            all_results = self.collection.get(
                include=["documents", "metadatas"]
            )

            if not all_results["ids"]:
                return {
                    "success": True,
                    "backup_data": json.dumps([]),
                    "total_memories": 0,
                    "backup_timestamp": datetime.utcnow().isoformat(),
                    "collection_name": self.collection_name
                }

            # Build backup data
            memories = []
            for i, memory_id in enumerate(all_results["ids"]):
                memories.append({
                    "id": memory_id,
                    "content": all_results["documents"][i],
                    "metadata": all_results["metadatas"][i]
                })

            backup_data = json.dumps(memories, indent=2, ensure_ascii=False)
            backup_timestamp = datetime.utcnow().isoformat()

            logger.info(
                f"✅ Backup created: {len(memories)} memories from collection '{self.collection_name}'"
            )

            return {
                "success": True,
                "backup_data": backup_data,
                "total_memories": len(memories),
                "backup_timestamp": backup_timestamp,
                "collection_name": self.collection_name
            }

        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            raise ValueError(f"Failed to create backup: {e}") from e

    def get_embedding(self, text: str) -> dict[str, Any]:
        """
        Get the embedding vector for a given text.

        Uses the same embedding model as the collection to generate
        an embedding vector for the provided text. Useful for debugging,
        analysis, or manual similarity calculations.

        Args:
            text: Text to generate embedding for

        Returns:
            dict: Embedding information
                {
                    "success": bool,
                    "text": str,
                    "embedding": list[float],  # The embedding vector
                    "dimension": int,  # Vector dimension
                    "model_info": str
                }
        """
        try:
            # Get embedding function from collection
            embedding_function = self.collection._embedding_function

            # Generate embedding
            embeddings = embedding_function([text])
            embedding = embeddings[0] if embeddings else []

            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            logger.info(f"✅ Generated embedding for text: '{text[:50]}...'")

            return {
                "success": True,
                "text": text,
                "embedding": embedding_list,
                "dimension": len(embedding_list),
                "model_info": "ChromaDB default embedding function"
            }

        except Exception as e:
            logger.error(f"❌ Failed to get embedding: {e}")
            raise ValueError(f"Failed to get embedding: {e}") from e

    def check_embedding_model(self) -> dict[str, Any]:
        """
        Get information about the embedding model used by the collection.

        Returns details about the embedding function, model configuration,
        and capabilities. Useful for debugging and understanding how
        semantic search works.

        Returns:
            dict: Model information
                {
                    "success": bool,
                    "model_name": str,
                    "embedding_dimension": int,
                    "collection_name": str,
                    "model_type": str,
                    "details": dict
                }
        """
        try:
            # Get embedding function
            embedding_function = self.collection._embedding_function

            # Get model info
            model_type = type(embedding_function).__name__

            # Test embedding to get dimension
            test_embedding = embedding_function(["test"])
            dimension = len(test_embedding[0]) if test_embedding else 0

            logger.info(f"✅ Embedding model info retrieved: {model_type}, dimension={dimension}")

            return {
                "success": True,
                "model_name": "ChromaDB Default Embedding Function",
                "embedding_dimension": dimension,
                "collection_name": self.collection_name,
                "model_type": model_type,
                "details": {
                    "description": "ChromaDB uses ONNX-based sentence transformers for embeddings",
                    "typical_model": "all-MiniLM-L6-v2 or similar",
                    "dimension": dimension,
                    "supports_multilingual": True
                }
            }

        except Exception as e:
            logger.error(f"❌ Failed to check embedding model: {e}")
            raise ValueError(f"Failed to check embedding model: {e}") from e

    def recall_by_timeframe(
        self,
        start_time: float,
        end_time: float,
        query: str | None = None,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories within a specific timeframe with optional semantic search.

        This is the explicit version of recall_memory - instead of natural language
        time parsing, you provide exact Unix timestamps for start and end times.

        Args:
            start_time: Start timestamp (Unix timestamp as float)
                       Example: 1730505600.0 (November 2, 2024 00:00:00 UTC)
            end_time: End timestamp (Unix timestamp as float)
                     Example: 1730592000.0 (November 3, 2024 00:00:00 UTC)
            query: Optional semantic search query to filter results within timeframe
            limit: Maximum number of results to return (default: 10)

        Returns:
            list[dict]: List of memories within the timeframe, optionally filtered by query

        Raises:
            ValueError: If start_time > end_time

        Examples:
            - recall_by_timeframe(1730505600.0, 1730592000.0)
              → All memories from Nov 2-3, 2024

            - recall_by_timeframe(time.time() - 86400, time.time(), query="Python")
              → Python-related memories from last 24 hours

        Notes:
            - Both start_time and end_time are inclusive
            - If query is provided, performs semantic search within timeframe
            - If no query, returns all memories in chronological order
            - Uses created_at timestamp for filtering
        """
        if start_time > end_time:
            raise ValueError("start_time must be less than or equal to end_time")

        try:
            # Build where clause for timeframe filtering
            where_clause = {
                "$and": [
                    {"created_at": {"$gte": start_time}},
                    {"created_at": {"$lte": end_time}}
                ]
            }

            logger.info(f"Recall by timeframe: {start_time} to {end_time}, query='{query}', limit={limit}")

            # If query provided, do semantic search within timeframe
            if query:
                results = self.search_memories(
                    query=query,
                    limit=limit,
                    filters=where_clause
                )
            # Otherwise, list all memories in timeframe
            else:
                results = self.list_memories(
                    limit=limit,
                    filters=where_clause
                )

            logger.info(f"✅ Recall by timeframe returned {len(results)} memories")
            return results

        except Exception as e:
            logger.error(f"❌ Failed to recall by timeframe: {e}")
            raise ValueError(f"Failed to recall by timeframe: {e}") from e

    def update_memory_metadata(
        self,
        memory_id: str,
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Update only the metadata of an existing memory without changing its content.

        This is a specialized update operation - unlike update_memory() which can change
        content, tags, and metadata, this method ONLY updates metadata fields. The memory
        content and tags remain unchanged.

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

        Raises:
            ValueError: If memory with given ID is not found

        Examples:
            - update_memory_metadata("abc-123", {"priority": "high"})
              → Adds/updates priority field, keeps all other metadata

            - update_memory_metadata("abc-123", {"reviewed": True, "reviewer": "Claude"})
              → Adds review metadata without touching content

        Notes:
            - Content and tags are NEVER modified by this operation
            - New metadata fields are added, existing fields are updated
            - updated_at timestamp is automatically updated
            - Use update_memory() if you need to change content or tags
        """
        try:
            # Get current memory
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"]
            )

            if not result["ids"]:
                raise ValueError(f"Memory with ID '{memory_id}' not found")

            # Get existing metadata
            current_metadata = result["metadatas"][0]

            # Merge new metadata with existing (new values override)
            updated_metadata = {**current_metadata, **metadata}

            # Always update the updated_at timestamp
            updated_metadata["updated_at"] = time.time()

            # Update in ChromaDB (content and embedding stay the same)
            self.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )

            updated_fields = list(metadata.keys())
            logger.info(f"✅ Updated metadata for memory {memory_id}: {updated_fields}")

            return {
                "success": True,
                "id": memory_id,
                "operation": "metadata_update",
                "updated_fields": updated_fields,
                "metadata": updated_metadata
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to update memory metadata: {e}")
            raise ValueError(f"Failed to update memory metadata: {e}") from e

    def delete_all_memories(self) -> bool:
        """
        Delete all memories from the collection.

        WARNING: This is destructive and cannot be undone!

        Returns:
            bool: True if successful
        """
        try:
            # Get all IDs
            all_results = self.collection.get()

            if all_results["ids"]:
                self.collection.delete(ids=all_results["ids"])
                logger.warning(f"⚠️  Deleted {len(all_results['ids'])} memories from collection")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete all memories: {e}")
            return False
