"""
retrieval_agent.py — Agent 2: Hybrid Search & Reranking
---------------------------------------------------------
Responsibility:
  Take the rewritten query from the Research Agent and retrieve the
  most relevant document chunks using the full hybrid pipeline:
  Dense → BM25 → RRF → Cross-Encoder Rerank

Input  (from LangGraph state): rewritten_query (str)
Output (into LangGraph state): reranked_chunks (list of dicts)

Each returned chunk has:
  {
    "text"         : str,
    "metadata"     : {"filename": str, "page": int, ...},
    "rerank_score" : float,
  }
"""

from rich.table import Table
from src.config import console, PREFETCH_K, FINAL_K
from src.retrieval.hybrid_retriever import retrieve


def run_retrieval_agent(state: dict) -> dict:
    """
    LangGraph node — runs hybrid retrieval and displays results.
    """
    console.rule("[agent]🔎  AGENT 2 — DATA RETRIEVAL AGENT[/agent]")

    query = state.get("rewritten_query", "").strip()

    # If the Research Agent refused the question, skip retrieval
    if query == "REFUSE" or not query:
        console.print("  [warning]⚠  Skipping retrieval (question was refused or empty)[/warning]")
        return {**state, "reranked_chunks": []}

    console.print(f"  [dim]📥 Query:[/dim] [data]{query!r}[/data]")
    chunks = retrieve(query, prefetch_k=PREFETCH_K, final_k=FINAL_K)

    # Pretty-print retrieved chunks in a table
    if chunks:
        table = Table(
            title="Retrieved Chunks (after reranking)",
            show_lines=True,
            style="dim",
            title_style="bold blue",
        )
        table.add_column("#",         style="bold", width=3)
        table.add_column("Score",     width=7)
        table.add_column("Source",    width=22)
        table.add_column("Page",      width=5)
        table.add_column("Preview",   width=55)

        for i, chunk in enumerate(chunks, 1):
            meta    = chunk["metadata"]
            preview = chunk["text"][:120].replace("\n", " ") + ("…" if len(chunk["text"]) > 120 else "")
            table.add_row(
                str(i),
                f"{chunk['rerank_score']:.3f}",
                meta.get("filename", "?"),
                str(meta.get("page", "?")),
                preview,
            )
        console.print(table)
    else:
        console.print("  [warning]⚠  No chunks retrieved — collection may be empty.[/warning]")

    return {**state, "reranked_chunks": chunks}

