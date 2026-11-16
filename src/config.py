"""
config.py — Central Configuration
-----------------------------------
Single place to change models, paths, chunk sizes, retrieval settings.
Every other module imports from here so changes propagate automatically.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme

# Load .env file automatically
load_dotenv()

# Suppress HuggingFace cache warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent          # project root
DATA_DIR        = BASE_DIR / "data"                     # put PDFs here
STORAGE_DIR     = BASE_DIR / "qdrant_storage"           # Qdrant local DB
BM25_CORPUS     = STORAGE_DIR / "bm25_corpus.json"      # BM25 index on disk

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

# ── Model Names ──────────────────────────────────────────────────────────────
# LLM: gemini-3.5-flash (free tier: fast, multimodal, 1M context)
GEMINI_LLM_MODEL    = "gemini-3.5-flash"

# Embeddings: gemini-embedding-001 (free, 3072-dim output)
GEMINI_EMBED_MODEL  = "gemini-embedding-001"
EMBED_DIMENSION     = 3072

# Reranker: local cross-encoder (free, ~90 MB, downloads on first use)
RERANK_MODEL        = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Qdrant ───────────────────────────────────────────────────────────────────
# In Docker/Kubernetes, set QDRANT_URL="http://qdrant:6333"
# If unset or empty, falls back to local embedded mode on disk
QDRANT_URL      = os.getenv("QDRANT_URL", "").strip()
QDRANT_PATH     = str(STORAGE_DIR)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag")

_qdrant_instance = None


def get_qdrant_client():
    """
    Returns a QdrantClient instance:
    - Remote HTTP client if QDRANT_URL is configured (Container / K8s mode)
    - Embedded local storage client singleton if QDRANT_URL is not set (Local dev mode)
    """
    global _qdrant_instance
    from qdrant_client import QdrantClient
    if QDRANT_URL == ":memory:":
        if _qdrant_instance is None:
            _qdrant_instance = QdrantClient(":memory:")
        return _qdrant_instance
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    if _qdrant_instance is None:
        _qdrant_instance = QdrantClient(path=QDRANT_PATH)
    return _qdrant_instance

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between adjacent chunks

# ── Retrieval ────────────────────────────────────────────────────────────────
PREFETCH_K = 20   # candidates from each retriever (dense + BM25)
FINAL_K    = 5    # chunks passed to the generation agent after reranking

# ── Embedding batching (to stay within Gemini rate limits) ───────────────────
EMBED_BATCH_SIZE   = 50     # chunks per API call
EMBED_BATCH_DELAY  = 1.0    # seconds between batches

# ── Rich Terminal Theme ───────────────────────────────────────────────────────
_theme = Theme({
    "info"    : "cyan",
    "success" : "bold green",
    "warning" : "bold yellow",
    "error"   : "bold red",
    "agent"   : "bold magenta",
    "step"    : "bold blue",
    "data"    : "white",
    "dim"     : "dim white",
})

console = Console(theme=_theme)

