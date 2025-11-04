# Multi-stage build - Context7 Best Practice
FROM python:3.13-slim AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./

# Install dependencies (frozen lockfile in production)
RUN uv venv && \
    uv pip install -e .

# Runtime stage - minimal image
FROM python:3.13-slim

# Context7 Security Best Practice: non-root user
RUN useradd -m -u 1000 appuser && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code with correct ownership
COPY --chown=appuser:appuser src/ ./src/

# Switch to non-root user
USER appuser

# Context7 Best Practice: Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8080

# Readiness-based healthcheck with startup grace period
# --interval: Check every 10 seconds (reasonable for production)
# --timeout: Single check timeout (3 seconds)
# --start-period: Grace period during initialization (120s to cover ONNX model loading ~90s)
# --retries: Number of consecutive failures before marking unhealthy (3)
#
# Readiness probe logic (/ready endpoint):
# - During ONNX loading: Returns 503 Service Unavailable → healthcheck fails (but not unhealthy during start-period)
# - When operational: Returns 200 OK with ready=true → healthcheck passes, container becomes "healthy"
# - On critical error: Returns 503 → healthcheck fails, container becomes "unhealthy" after retries
#
# For liveness-only checks (process alive), use /health endpoint instead (always returns 200 OK)
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ready || exit 1

# Run MCP server directly
CMD ["python", "-m", "src.server"]
