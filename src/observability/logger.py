"""
logger.py — Centralized Structured JSON Logging
------------------------------------------------
Outputs structured JSON logs to stdout in production, enabling seamless ingestion
by log aggregation platforms (Loki, Elasticsearch, Datadog, AWS CloudWatch).
"""

import os
import sys
import logging
try:
    from pythonjsonlogger import json as jsonlogger
except ImportError:
    from pythonjsonlogger import jsonlogger


def setup_logging(default_level: str = "INFO"):
    """
    Configure global logging.
    If LOG_FORMAT=json or running in container, formats output as JSON.
    Otherwise uses standard formatted text.
    """
    log_level = os.getenv("LOG_LEVEL", default_level).upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if log_format == "json":
        # Cloud-native structured JSON format
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    else:
        # Human-readable format
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the specified module."""
    return logging.getLogger(name)
