"""
embedder.py — Gemini Query Embedder
--------------------------------------
Embeds user queries (not documents — those are embedded at ingestion time)
using Gemini text-embedding-004 via the new google.genai SDK.

We use task_type="retrieval_query" for queries vs "retrieval_document"
for documents. This tells the model to optimize the vector for search.
"""

from google import genai
from src.config import GEMINI_API_KEY, GEMINI_EMBED_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


def embed_query(query: str) -> list[float]:
    """
    Convert a query string into a 768-dimensional embedding vector.

    Uses task_type="retrieval_query" (different from document embeddings)
    so the model knows this vector will be used to SEARCH, not store.

    Returns:
        list[float] of length 768
    """
    response = _client.models.embed_content(
        model   = GEMINI_EMBED_MODEL,
        contents= [query],
        config  = {"task_type": "retrieval_query"},
    )
    if not response.embeddings:
        raise RuntimeError(f"Gemini returned no embedding for query: {query!r}")
    return response.embeddings[0].values
