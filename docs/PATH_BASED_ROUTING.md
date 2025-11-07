# Path-Based Routing - Multi-Project Setup

Yggdrasil supports **path-based routing** - dynamic collection selection based on URL path. This enables **one Docker container** to serve multiple Claude Code projects simultaneously!

## How It Works

### URL Pattern:

```
http://localhost:8080/mcp-{project_name}
```

### Automatic Mapping:

```
/mcp-enigma     →  collection: "enigma_memories"
/mcp-alpha      →  collection: "alpha_memories"
/mcp-yggdrasil  →  collection: "yggdrasil_memories"
/mcp            →  collection: "default_memories" (from CHROMA_COLLECTION env var)
```

## Per-Project Configuration

### Project A (e.g., "Enigma")

File: `/path/to/project-enigma/.mcp.json`

```json
{
  "mcpServers": {
    "Yggdrasil": {
      "type": "http",
      "url": "http://localhost:8080/mcp-enigma",
      "metadata": {
        "description": "Yggdrasil MCP Memory Server"
      }
    }
  }
}
```

### Project B (e.g., "Alpha")

File: `/path/to/project-alpha/.mcp.json`

```json
{
  "mcpServers": {
    "Yggdrasil": {
      "type": "http",
      "url": "http://localhost:8080/mcp-alpha",
      "metadata": {
        "description": "Yggdrasil MCP Memory Server"
      }
    }
  }
}
```

### Project C (e.g., "Yggdrasil Development")

File: `/path/to/yggdrasil/.mcp.json`

```json
{
  "mcpServers": {
    "Yggdrasil": {
      "type": "http",
      "url": "http://localhost:8080/mcp-yggdrasil",
      "metadata": {
        "description": "Yggdrasil MCP Memory Server"
      }
    }
  }
}
```

## Benefits

- **ONE Docker container** instead of N containers
- **ONE ONNX model in memory** (~80MB) instead of N×80MB
- **Zero rebuilds** when adding new projects
- **Automatic collection creation** in Chroma Cloud
- **Memory isolation** between projects

## Architecture

```
┌──────────────────┐
│  Project Enigma  │
│  .mcp.json:      │──┐
│  /mcp-enigma     │  │
└──────────────────┘  │
                      │   ┌────────────────────┐       ┌───────────────────────────┐
┌──────────────────┐  ├──>│  Docker Yggdrasil  │──────>│  Chroma Cloud collections │
│   Project Alpha  │  │   │  Port 8080         │       │                           │
│  .mcp.json:      │──┤   │  Path Router       │       │  • enigma_memories        │
│  /mcp-alpha      │  │   └────────────────────┘       │  • alpha_memories         │
└──────────────────┘  │                                │  • yggdrasil_memories     │
                      │                                └───────────────────────────┘
┌──────────────────┐  │
│ Project Yggdrasil│  │
│  .mcp.json:      │──┘
│  /mcp-yggdrasil  │
└──────────────────┘
```

## Implementation

### 1. Request Context (contextvars)

Each request has its own context with `collection_name`.

### 2. Middleware Routing

1. **Parses**: `/mcp-enigma` → collection_name = "enigma_memories"
2. **Rewrite**: `/mcp-enigma` → `/mcp` (for FastMCP routing)
3. **Context**: set_collection_name("enigma_memories")

### 3. Dynamic Collection Selection

**get_collection()** resolution order:

1. Explicit parameter
2. Request context (from URL path)
3. Default from CHROMA_COLLECTION env var

Use dashboard or Chroma Cloud API.

## Example Flow

1. **Claude Code** (Enigma Project) sends request → `POST http://localhost:8080/mcp-enigma`
2. **Middleware** parses path → `collection_name = "enigma_memories"`
3. **Context** sets collection name for this request
4. **Path rewrite**: `/mcp-enigma` → `/mcp` (for FastMCP)
5. **MCP Handler** receives request
6. **Tool** calls `get_collection()` → reads `enigma_memories` from context
7. **Chroma Client** uses collection `enigma_memories`
8. **Response** returns to Claude Code

## FAQ

<details>
<summary><strong>What happens if I forget the suffix in URL?</strong></summary>

`/mcp` will use the default collection from `CHROMA_COLLECTION` env var from `.env` file.

</details>

<details>
<summary><strong>Do I need to change anything in docker-compose.yml?</strong></summary>

No! One container on port `8080` is enough.

</details>

<details>
<summary><strong>How are collections named?</strong></summary>

`/mcp-{name}` → `{name}_memories` (automatically)

</details>

<details>
<summary><strong>Can I use special characters in project name?</strong></summary>

Use only `[a-zA-Z0-9_-]`. Other characters will be replaced with `_`.

</details>

<details>
<summary><strong>Are collections created automatically?</strong></summary>

Yes! On first request, Yggdrasil will create the collection in Chroma Cloud.

</details>
