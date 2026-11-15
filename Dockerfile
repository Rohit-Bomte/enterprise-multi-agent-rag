# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Production Enterprise Dockerfile for Mini Enterprise AI Knowledge Assistant
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim as base

# Set environment variables for production Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    LOG_FORMAT=json

WORKDIR /app

# Install OS-level dependencies (curl for healthchecks, build-essential if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user and data directory
RUN mkdir -p /app/data && \
    groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the Cross-Encoder model into the image to ensure sub-second pod cold starts
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application source code and static assets
COPY src/ ./src/
COPY static/ ./static/

# Set ownership to unprivileged user
RUN chown -R appuser:appgroup /app /home/appuser

# Switch to non-root user
USER appuser

# Expose port for FastAPI application
EXPOSE 8000

# Container healthcheck targeting the Kubernetes/Docker health endpoint
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
