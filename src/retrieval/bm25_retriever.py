"""
bm25_retriever.py — Keyword Search with BM25
----------------------------------------------
BM25 (Best Match 25) is a classic information-retrieval algorithm that
ranks documents by how often the query's words appear in them.

It is fast and extremely good at exact-term matching — names, codes,
technical identifiers, etc. — where dense embedding search may miss.

We load the BM25 corpus that was saved to disk during ingestion, build
a BM25 index in memory, and return the top-k matching chunks.
"""

import json
from rank_bm25 import BM25Okapi
from src.config import BM25_CORPUS, console


def _load_corpus() -> list[dict]:
    """Load the BM25 corpus saved during ingestion."""
    if not BM25_CORPUS.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {BM25_CORPUS}. "
            "Run the ingestion pipeline first:\n"
            "  python -m src.ingest_pipeline <path/to/file.pdf>"
        )
    with open(BM25_CORPUS, "r", encoding="utf-8") as f:
        return json.load(f)


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """
    Search the BM25 corpus for the top-k most keyword-relevant chunks.

    Args:
        query: The search query string
        top_k: Number of top results to return

    Returns:
        list of dicts: [{"text": str, "metadata": dict, "bm25_score": float, "bm25_rank": int}]

    How it works:
        1. Tokenize query and all corpus documents (simple whitespace split)
        2. BM25Okapi computes a relevance score for each document
        3. Return the top-k documents sorted by score descending
    """
    corpus = _load_corpus()
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Sort by score and pick top_k
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices):
        if scores[idx] > 0:   # only include documents with any keyword match
            results.append({
                "text"       : corpus[idx]["text"],
                "metadata"   : corpus[idx]["metadata"],
                "bm25_score" : float(scores[idx]),
                "bm25_rank"  : rank,
                "corpus_id"  : idx,
            })

    console.print(
        f"  [dim]BM25:[/dim] [bold]{len(results)}[/bold] keyword matches "
        f"[dim](from {len(corpus)} total chunks)[/dim]"
    )
    return results

