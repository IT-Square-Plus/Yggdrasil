# 📖 MCP Tools Reference

Yggdrasil provides **28 MCP tools** organized by category:

## Core Memory Operations

### `save_memory` - Store new memory

```json
{
  "content": "Python 3.13 released with improved type hints",
  "tags": ["python", "release"],
  "metadata": { "source": "official docs", "priority": "high" }
}
```

### `search_memories` - Semantic search

```json
{
  "query": "What's new in Python 3.13?",
  "limit": 10,
  "filters": { "tags": "python" }
}
```

### `get_memory` - Retrieve by ID

```json
{
  "memory_id": "uuid-here"
}
```

### `update_memory` - Update existing memory

```json
{
  "memory_id": "uuid-here",
  "content": "Updated content",
  "tags": ["updated", "python"]
}
```

### `delete_memory` - Delete by ID

```json
{
  "memory_id": "uuid-here"
}
```

### `list_memories` - List with pagination

```json
{
  "limit": 50,
  "offset": 0,
  "filters": { "tags": "important" }
}
```

### `get_memory_stats` - Get collection statistics

```json
{}
```

## Time-Based Retrieval

### `recall_memory` - Natural language time queries

```json
{
  "query": "what did I save yesterday about Python?"
}
```

Supports: "yesterday", "last week", "2 days ago", "this morning", "recent"

### `recall_by_timeframe` - Explicit timestamp range

```json
{
  "start_time": 1730505600.0,
  "end_time": 1730592000.0,
  "query": "Python"
}
```

## Tag-Based Search

### `search_by_tag` - ANY tag matching (OR logic)

```json
{
  "tags": ["python", "javascript"]
}
```

### `search_by_all_tags` - ALL tags matching (AND logic)

```json
{
  "tags": ["python", "test", "production"]
}
```

## Advanced Operations

### `merge_memories` - Combine multiple memories

```json
{
  "memory_ids": ["id1", "id2", "id3"],
  "separator": "\n\n---\n\n",
  "delete_originals": false
}
```

### `update_memory_tags` - Tag management

```json
{
  "memory_id": "uuid-here",
  "tags": ["important"],
  "mode": "append" // or "replace", "remove"
}
```

### `update_memory_metadata` - Update metadata only

```json
{
  "memory_id": "uuid-here",
  "metadata": { "priority": "high", "reviewed": true }
}
```

### `update_collection_metadata` - Update Chroma collection metadata

```json
{
  "metadata": "{\"project\": \"Enigma AI\", \"owner\": \"Łukasz\", \"version\": \"1.0.0\"}"
}
```

Accepts JSON or YAML format.

## Export & Import

### `export_memories` - Export to JSON/Markdown/CSV

```json
{
  "format": "json", // or "markdown", "csv"
  "filters": { "tags": "python" },
  "include_metadata": true
}
```

### `import_memories` - Import from backup

```json
{
  "data": "json_string_here",
  "format": "json",
  "preserve_ids": false,
  "on_duplicate": "skip" // or "overwrite", "error"
}
```

### `create_backup` - Full collection backup

```json
{}
```

## Maintenance

### `cleanup_duplicates` - Remove exact duplicates

```json
{
  "similarity_threshold": 0.95
}
```

### `optimize_db` - Database optimization

```json
{}
```

### `check_health` - System health check

```json
{}
```

## Bulk Deletion

### `delete_by_tag` - Delete by ANY tag

```json
{
  "tags": ["temporary", "draft"]
}
```

### `delete_by_all_tags` - Delete by ALL tags

```json
{
  "tags": ["test", "archived"]
}
```

### `delete_by_timeframe` - Delete within time range

```json
{
  "start_time": 1730505600.0,
  "end_time": 1730592000.0
}
```

### `delete_before_date` - Delete older than date

```json
{
  "timestamp": 1730505600.0
}
```

## Debug Tools

### `debug_retrieve` - Debug semantic search

```json
{
  "query": "Python",
  "limit": 10,
  "similarity_threshold": -1.0
}
```

### `get_embedding` - Get embedding vector

```json
{
  "text": "machine learning"
}
```

### `check_embedding_model` - Model information

```json
{}
```

---

**For usage examples in Claude Code, see main README.md**
