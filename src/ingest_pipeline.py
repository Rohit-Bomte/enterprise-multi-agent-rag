"""
ingest_pipeline.py — CLI: Ingest PDF(s) into the RAG System
-------------------------------------------------------------
Run this script once for each PDF you want the system to know about.

Usage:
    python -m src.ingest_pipeline data/my_report.pdf
    python -m src.ingest_pipeline data/report1.pdf data/report2.pdf
    python -m src.ingest_pipeline data/my_report.pdf --reset   # wipe & re-index

What happens internally (all visible in terminal):
  [1/4] Parse   — PyMuPDF extracts text page by page
  [2/4] Sanitize — Remove embedded prompt injections from document text
  [3/4] Chunk   — Split pages into 512-char overlapping segments
  [4/4] Index   — Embed with Gemini + store in Qdrant + save BM25 corpus
"""

import sys
import argparse
from pathlib import Path

from rich.panel import Panel

from src.config import console
from src.ingestion.parser    import parse_pdf
from src.ingestion.chunker   import chunk_pages
from src.ingestion.sanitizer import sanitize_chunks
from src.ingestion.indexer   import index_chunks


def ingest_file(pdf_path: str, reset: bool = False):
    """Full ingestion pipeline for a single PDF file."""
    path = Path(pdf_path)

    console.print(Panel(
        f"[bold]Ingesting:[/bold] [data]{path.name}[/data]",
        title="📄 PDF RAG — Ingestion Pipeline",
        border_style="blue",
    ))

    # ── Step 1: Parse ───────────────────────────────────────────────────────
    console.print("\n[step]  ━━━  STEP 1 / 4  —  PDF PARSING  ━━━[/step]")
    pages = parse_pdf(  str(path))

    # ── Step 2: Sanitize ────────────────────────────────────────────────────
    console.print("\n[step]  ━━━  STEP 2 / 4  —  DOCUMENT SANITIZATION  ━━━[/step]")
    console.print("  [dim]Scanning for embedded prompt injections…[/dim]")
    # Build temporary chunks to sanitize at page level, then rechunk
    temp_chunks = [{"text": p["text"], "metadata": p["metadata"]} for p in pages]
    temp_chunks = sanitize_chunks(temp_chunks)
    # Put sanitized text back into pages
    for page, chunk in zip(pages, temp_chunks):
        page["text"] = chunk["text"]

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    console.print("\n[step]  ━━━  STEP 3 / 4  —  CHUNKING  ━━━[/step]")
    chunks = chunk_pages(pages)

    # ── Step 4: Index ─────────────────────────────────────────────────────────
    console.print("\n[step]  ━━━  STEP 4 / 4  —  EMBEDDING & INDEXING  ━━━[/step]")
    index_chunks(chunks, reset=reset)

    console.print(Panel(
        f"[success]✔  '{path.name}' is ready for queries![/success]\n"
        f"[dim]Run:[/dim]  python -m src.main",
        title="✅ Ingestion Complete",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into the RAG system."
    )
    parser.add_argument(
        "pdf_files",
        nargs="+",
        help="Path(s) to PDF file(s) to ingest",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing collection and re-index from scratch",
    )
    args = parser.parse_args()

    for pdf_path in args.pdf_files:
        ingest_file(pdf_path, reset=args.reset)


if __name__ == "__main__":
    main()

