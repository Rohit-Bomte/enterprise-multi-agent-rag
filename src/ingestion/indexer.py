"""
indexer.py — Embed Chunks and Store in Qdrant
----------------------------------------------
Pipeline:
  chunks → Gemini text-embedding-004 → 768-dim vectors → Qdrant (local)
  chunks → plain text corpus → bm25_corpus.json (for keyword search)

Qdrant runs in LOCAL embedded mode — no Docker or server required.
Data persists in the ./qdrant_storage folder.
"""

import json
import time
from google import genai as _genai_module

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.config import (
    console, GEMINI_API_KEY, GEMINI_EMBED_MODEL, EMBED_DIMENSION,
    QDRANT_PATH, COLLECTION_NAME, BM25_CORPUS,
    EMBED_BATCH_SIZE, EMBED_BATCH_DELAY, get_qdrant_client,
)

# Configure Gemini client (new google.genai SDK)
_genai_client = _genai_module.Client(api_key=GEMINI_API_KEY)


# ── Qdrant Client ─────────────────────────────────────────────────────────────
def _get_qdrant() -> QdrantClient:
    return get_qdrant_client()


def create_collection(reset: bool = False):
    """Create (or recreate) the Qdrant collection for dense vectors."""
    client = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if reset:
            client.delete_collection(COLLECTION_NAME)
            console.print(f"  [warning]⚠  Deleted existing collection '{COLLECTION_NAME}'[/warning]")
        else:
            console.print(
                f"  [info]ℹ  Collection '{COLLECTION_NAME}' already exists — "
                "appending. Use --reset to start fresh.[/info]"
            )
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIMENSION, distance=Distance.COSINE),
    )
    console.print(f"  [success]✔[/success]  Created collection [bold]{COLLECTION_NAME}[/bold]  "
                  f"[dim](dim={EMBED_DIMENSION}, metric=cosine)[/dim]")


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using Gemini text-embedding-004.
    Returns list of 768-dim float vectors.
    Uses the new google.genai SDK.
    """
    response = _genai_client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=texts,
        config={"task_type": "retrieval_document"},
    )
    # response.embeddings is a list of ContentEmbedding objects
    return [emb.values for emb in response.embeddings]


# ── Main Indexing Function ────────────────────────────────────────────────────
def index_chunks(chunks: list[dict], reset: bool = False):
    """
    Embed all chunks and store them in Qdrant + save BM25 corpus to disk.

    Args:
        chunks: Output from chunker.chunk_pages()
        reset:  If True, wipe the existing collection first
    """
    if not chunks:
        console.print("  [warning]No chunks to index.[/warning]")
        return

    create_collection(reset=reset)
    client = _get_qdrant()

    texts = [c["text"] for c in chunks]
    total = len(texts)
    all_embeddings: list[list[float]] = []

    console.print(
        f"  [info]ℹ  Embedding {total} chunks in batches of {EMBED_BATCH_SIZE} "
        f"(model: {GEMINI_EMBED_MODEL})…[/info]"
    )

    with Progress(
        TextColumn("  [step]  Embedding[/step]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} chunks)"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=total)
        for i in range(0, total, EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            embeddings = embed_texts(batch)
            all_embeddings.extend(embeddings)
            progress.advance(task, len(batch))
            if i + EMBED_BATCH_SIZE < total:
                time.sleep(EMBED_BATCH_DELAY)   # respect free-tier rate limit

    # Build Qdrant points
    # Determine starting ID so we can append without overwriting
    existing_count = client.count(collection_name=COLLECTION_NAME).count
    points = [
        PointStruct(
            id      = existing_count + i,
            vector  = embedding,
            payload = {"text": chunk["text"], **chunk["metadata"]},
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings))
    ]

    # Upsert in batches of 100
    console.print(f"  [info]ℹ  Upserting {len(points)} points into Qdrant…[/info]")
    for i in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + 100])

    # Save BM25 corpus to disk
    _save_bm25_corpus(chunks)

    total_stored = client.count(collection_name=COLLECTION_NAME).count
    console.print(
        f"  [success]✔[/success]  Qdrant: [bold]{total_stored}[/bold] vectors stored  "
        f"[dim](path: {QDRANT_PATH})[/dim]"
    )


def _save_bm25_corpus(chunks: list[dict]):
    """
    Append chunk texts to the BM25 corpus JSON file.
    This file is used by the BM25 retriever at query time.
    """
    # Load existing corpus if present
    existing = []
    if BM25_CORPUS.exists():
        with open(BM25_CORPUS, "r", encoding="utf-8") as f:
            existing = json.load(f)

    new_entries = [
        {"text": c["text"], "metadata": c["metadata"]}
        for c in chunks
    ]
    combined = existing + new_entries

    with open(BM25_CORPUS, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    console.print(
        f"  [success]✔[/success]  BM25 corpus saved  "
        f"[dim]({len(combined)} total documents → {BM25_CORPUS.name})[/dim]"
    )

