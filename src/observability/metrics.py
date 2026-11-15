"""
metrics.py — Prometheus Metrics for Multi-Agent RAG
---------------------------------------------------
Exposes custom domain metrics alongside standard HTTP metrics for Prometheus scraping.
"""

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# ── Custom Application Metrics ────────────────────────────────────────────────

# Total user queries processed by status
rag_query_counter = Counter(
    name="rag_queries_total",
    documentation="Total number of RAG queries processed",
    labelnames=["status"],  # "success", "refused", "error"
)

# Latency distribution for end-to-end RAG queries
rag_query_latency = Histogram(
    name="rag_query_duration_seconds",
    documentation="Latency of full RAG query execution in seconds",
    buckets=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0],
)

# Latency for individual multi-agent steps
rag_agent_latency = Histogram(
    name="rag_agent_step_duration_seconds",
    documentation="Latency per agent in the multi-agent pipeline",
    labelnames=["agent"],  # "research", "retrieval", "generation"
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Guardrail interception counter
rag_guardrail_blocks = Counter(
    name="rag_guardrail_blocks_total",
    documentation="Total count of security guardrail interventions",
    labelnames=["layer", "reason"],  # layer: input/output, reason: injection/length/pii/leakage
)

# Document ingestion metrics
rag_chunks_indexed = Counter(
    name="rag_chunks_indexed_total",
    documentation="Total number of document chunks indexed into the vector DB",
    labelnames=["source_file"],
)


def init_metrics(app):
    """
    Instrument the FastAPI application to expose Prometheus metrics at /metrics.
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/healthz"],
    )
    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics")
    return instrumentator

