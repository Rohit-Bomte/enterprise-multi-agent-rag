"""
hybrid_retriever.py — Dense + BM25 + Reranking
-------------------------------------------------
Combines three retrieval strategies for maximum quality:

  Step 1 — DENSE SEARCH
    Embed the query with Gemini → cosine similarity vs Qdrant vectors
    → top-20 candidates by semantic meaning

  Step 2 — BM25 KEYWORD SEARCH
    Score all corpus chunks by keyword frequency (rank_bm25)
    → top-20 candidates by exact term match

  Step 3 — RRF FUSION (Reciprocal Rank Fusion)
    Merge the two ranked lists by RANK POSITION, not raw scores
    (scores are on different scales, ranks are always comparable)
    → deduplicated merged list of up to 20 unique candidates

  Step 4 — CROSS-ENCODER RERANKING
    A local cross-encoder (ms-marco-MiniLM-L-6-v2) reads the full
    query + each candidate chunk TOGETHER and assigns a precise score
    → final top-5 chunks with the highest relevance
"""

from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # suppress HuggingFace warning

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from src.config import (
    console, QDRANT_PATH, COLLECTION_NAME,
    PREFETCH_K, FINAL_K, RERANK_MODEL, get_qdrant_client,
)
from src.retrieval.embedder import embed_query
from src.retrieval.bm25_retriever import bm25_search

# Load cross-encoder (prefer local cache to avoid network check on startup)
console.print(f"  [dim]Loading reranker ({RERANK_MODEL})…[/dim]")
try:
    _reranker = CrossEncoder(RERANK_MODEL, local_files_only=True)
except Exception:
    _reranker = CrossEncoder(RERANK_MODEL)
console.print("  [success]✔[/success]  Reranker ready")


def _dense_search(query_vector: list[float], top_k: int) -> list[dict]:
    """Semantic nearest-neighbour search in Qdrant."""
    client = get_qdrant_client()
    response = client.query_points(
        collection_name = COLLECTION_NAME,
        query           = query_vector,
        limit           = top_k,
        with_payload    = True,
    )
    return [
        {
            "text"        : r.payload["text"],
            "metadata"    : {k: v for k, v in r.payload.items() if k != "text"},
            "dense_score" : r.score,
            "dense_rank"  : i,
            "id"          : r.id,
        }
        for i, r in enumerate(response.points)
    ]


def _rrf_merge(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF):
        score(doc) = Σ  1 / (k + rank_in_list)

    k=60 is the standard default from the original RRF paper.
    Higher k = less weight on top-ranked items, more diversity.
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    def _key(chunk: dict) -> str:
        """Unique key: source file + page + first 60 chars of text."""
        m = chunk["metadata"]
        return f"{m.get('filename','?')}__p{m.get('page','?')}__{chunk['text'][:60]}"

    for rank, chunk in enumerate(dense_results):
        key = _key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        docs[key]   = chunk

    for rank, chunk in enumerate(bm25_results):
        key = _key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in docs:
            docs[key] = chunk

    merged = sorted(docs.keys(), key=lambda k: scores[k], reverse=True)
    return [docs[key] for key in merged]


def retrieve(query: str, prefetch_k: int = PREFETCH_K, final_k: int = FINAL_K) -> list[dict]:
    """
    Full hybrid retrieval pipeline for one query.

    Args:
        query:      Rewritten query string from the Research Agent
        prefetch_k: Candidates from each retriever (dense + BM25)
        final_k:    Final chunks after reranking

    Returns:
        list of chunk dicts with keys: text, metadata, rerank_score
    """
    console.print(f"\n  [dim]Query:[/dim] [italic]{query!r}[/italic]")

    # ── Step 1: Dense search ─────────────────────────────────────────────────
    console.print(f"  [step]  ① Dense search[/step]  (semantic, Qdrant + Gemini embeddings)…")
    query_vec     = embed_query(query)
    dense_results = _dense_search(query_vec, top_k=prefetch_k)
    console.print(
        f"     → [bold]{len(dense_results)}[/bold] candidates  "
        f"[dim](top score: {dense_results[0]['dense_score']:.3f} if any)[/dim]"
        if dense_results else "     → 0 candidates"
    )

    # ── Step 2: BM25 search ──────────────────────────────────────────────────
    console.print(f"  [step]  ② BM25 search[/step]   (keyword, rank_bm25)…")
    bm25_results = bm25_search(query, top_k=prefetch_k)

    # ── Step 3: RRF fusion ───────────────────────────────────────────────────
    console.print(f"  [step]  ③ RRF fusion[/step]    (merging ranked lists)…")
    merged = _rrf_merge(dense_results, bm25_results)
    candidates = merged[:prefetch_k]
    console.print(f"     → [bold]{len(candidates)}[/bold] unique candidates after fusion")

    if not candidates:
        console.print("  [warning]⚠  No candidates found. Is the collection empty?[/warning]")
        return []

    # ── Step 4: Cross-encoder reranking ──────────────────────────────────────
    console.print(f"  [step]  ④ Reranking[/step]     (cross-encoder: {RERANK_MODEL})…")
    pairs  = [(query, c["text"]) for c in candidates]
    scores = _reranker.predict(pairs)   # shape: (len(candidates),)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:final_k]

    score_strs = ", ".join(f"{c['rerank_score']:.3f}" for c in reranked)
    console.print(
        f"     → Top [bold]{len(reranked)}[/bold] chunks selected  "
        f"[dim](scores: [{score_strs}])[/dim]"
    )
    return reranked

