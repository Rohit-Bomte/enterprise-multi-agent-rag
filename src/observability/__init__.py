"""
Observability Package — Centralized Logging and Prometheus Metrics
"""
from src.observability.logger import get_logger, setup_logging
from src.observability.metrics import (
    rag_query_counter,
    rag_query_latency,
    rag_guardrail_blocks,
    rag_chunks_indexed,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "rag_query_counter",
    "rag_query_latency",
    "rag_guardrail_blocks",
    "rag_chunks_indexed",
]

