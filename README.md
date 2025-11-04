# 🌳 Yggdrasil MCP Memory Server

Production-ready MCP memory server with Chroma Cloud vector storage.

A modern Model Context Protocol (MCP) server that provides persistent semantic memory using Chroma Cloud vector database. Built with Python, FastAPI, and following Context7 best practices.

---

## ToC

1. [Features](#features)
2. [Architecture](#architecture)

   - [Technology Stack](#technology-stack)
   - [Project Structure](#project-structure)

3. [Quick Start](#quick-start)

   - [Prerequisites](#prerequisites)

     - [Get Chroma Cloud Credentials](#1-get-chroma-cloud-credentials)
     - [Configure Environment](#2-configure-environment)
     - [Run with Docker](#3-run-with-docker)

4. [ToDo](#todo)
5. [Extras](#extras)
6. [FAQ](#faq)
7. [Credits](#credits)
8. [Authors](#authors)

---

## Features

- **MCP Protocol 2025-03-26** - Streamable HTTP transport (stateless)
- **Chroma Cloud** - CloudClient integration (recommended by Context7)
- **Path-Based Routing**<sup>[1](#footnote-path-based-routing)</sup> - Multi-project support with ONE Docker container
- **Semantic Search** - Vector-based similarity search with ONNX embeddings<sup>[2](#footnote-onnx-model)</sup>
- **Tag Management** - Organize memories with custom tags and metadata
- **Time-Based Queries** - Natural language time expressions ("yesterday", "last week")
- **Docker Ready** - Multi-stage build, non-root user, proper healthchecks
- **Production Best Practices** - Context7 validated patterns throughout
- **28 MCP Tools** - Complete CRUD, search, export, merge, cleanup operations

---

## Architecture

### Technology Stack

- **Python 3.13**
- **MCP Server Stack:**
  - **FastMCP** - MCP protocol implementation (2025-03-26)
  - **FastAPI** - Async web framework
  - **Uvicorn** - ASGI server
- **Chroma Cloud** - Vector database for embeddings
- **Docker** - Containerization platform

### Project Structure

```
project-directory/
├── docs/                       # Documentation
├── [deployment files]          # docker-compose.yml, Dockerfile, pyproject.toml
├── .env.example                # Configuration template
│
└── src/
    ├── server.py               # MCP server stack
    ├── config/                 # Application settings
    │   ├── settings.py
    │   ├── chroma_client.py
    │   └── request_context.py
    ├── services/               # Business logic
    │   └── memory_service.py
    └── utils/                  # Helper functions
        └── time_parser.py
```

---

## Quick Start

### Prerequisites

- **Docker** (Engine or Desktop)
- **Chroma Cloud Account** - [https://www.trychroma.com](https://www.trychroma.com)
- **MCP Client** - Claude Code (recommended) or Claude Desktop (not tested)

### 1. Get Chroma Cloud Credentials

1. Create account Chroma Cloud<sup>[3](#footnote-chroma-cloud)</sup> at [https://www.trychroma.com](https://www.trychroma.com)
2. Create a new **Tenant** (your organisation)
3. Create a **Database** in your tenant
4. Generate an **API Key**
5. Note down: `tenant`, `database`, `api_key`

### 2. Configure Environment

```bash
# Clone this repository
git clone https://github.com/IT-Square-Plus/yggdrasil.git

# Navigate to project directory
cd yggdrasil

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required variables:**

```bash
CHROMA_API_KEY=your_api_key_here
CHROMA_TENANT=your_tenant_name
CHROMA_DATABASE=your_database_name
CHROMA_COLLECTION=default_memories  # Used only when URL is /mcp (not /mcp-project_name)
```

### 3. Run with Docker

```bash
# Build and start
docker compose up --build -d

# Check logs (wait for ONNX model download ~79MB, ~20 seconds)
docker compose logs -f
# You'll see: "🤖 ONNX embedding model successfully downloaded!"

# Check health
curl http://localhost:8080/ready
# Or open in browser: http://localhost:8080/ready
# If you see JSON response, MCP server is running and ONNX model loaded!
```

**Note:** First startup downloads ONNX model<sup>[2](#footnote-onnx-model)</sup> (~79MB, ~20 seconds) from Hugging Face and caches it locally.

---

## ToDo

- Claude Desktop compatibility (might work, haven't tested it yet)
- [WiP] Document/file parsing
- Gemini CLI compatibility

---

## Extras

If you're wondering about what the Chroma Cloud Quotas (Org Settings -> Quotas) and its values mean?

Have a look at this guide: [`docs/CHROMA_QUOTAS.md`](docs/CHROMA_QUOTAS.md)

---

## FAQ

<details>
<summary><strong>Does it work on Open AI's Codex?</strong></summary>

Not sure. Haven't tested and have no intention to work on Codex's compatibility yet. It's the lowest priotity now for me.

Might work out-of-the-box, might not. Not a Codex user, sorry.

</details>

<details>
<summary><strong>Why only Chroma Cloud and not self-hosted ChromaDB?</strong></summary>

I actually developed previously Yggdrasil using self-hosted ChromaDB 0.6.x so you could run it on your own server.

But then ChromaDB Devs released new version of ChromaDB rewritten in Rust and they haven't implemented Auth mechanism yet.

I was kind of too lazy to implement any sort of OAuth for 1.x and then I've discovered that they're providing Chroma Cloud.

It's damn cheap. No monthly contract, just pay-as-you-go model. You pay for what you use and API Auth included so why not?

</details>

---

## Credits

- Heavily inspired by Henry "[doobidoo](https://github.com/doobidoo "Heinrich Krupp")" Krupp's [MCP Memory Service](https://github.com/doobidoo/mcp-memory-service) project.

- Special thanks to [Jakub Gościniak](https://github.com/jgosciniak) for inspiring this project through our discussions about separate MCP Memory.

- Also thanks to [Maciej Waśniewski](https://github.com/maciejwasniewski) thanks to whom and our shared memory, we can work on the same project.

## Authors

- [Łukasz Bryzek](https://github.com/lukaszbryzek)

## 📚 Footnotes

### <a name="footnote-path-based-routing">[1]</a> Path-Based Routing

Enables **one** Yggdrasil Docker container to serve multiple Claude Code projects simultaneously, each with isolated memory collections.

For complete setup guide with examples, see: [`docs/PATH_BASED_ROUTING.md`](docs/PATH_BASED_ROUTING.md)

### <a name="footnote-onnx-model">[2]</a> ONNX Model

**Open Neural Network Exchange (ONNX)** is ChromaDB's AI engine for semantic search - it converts text into 384/768/1536/3072/4096-dimensional vectors representing meaning, enabling search by concept rather than keywords.

For detailed technical explanation, see: [`docs/ONNX.md`](docs/ONNX.md)

### <a name="footnote-chroma-cloud">[3]</a> Chroma Cloud

For full guide on how to set up and configure Chroma Cloud account, see: [`docs/CHROMA_CLOUD.md`](docs/CHROMA_CLOUD.md)
