"""
main.py — Interactive Q&A CLI
------------------------------
Run this after ingesting at least one PDF.

Usage:
    python -m src.main                        # interactive REPL
    python -m src.main -q "What is XYZ?"     # single question mode

The terminal output shows EVERY step of the pipeline so you can see
exactly what each agent and guardrail is doing internally.
"""

import argparse
from qdrant_client import QdrantClient
from rich.panel import Panel
from rich.rule import Rule

from src.config import console, QDRANT_PATH, COLLECTION_NAME
from src.workflow import ask


def _collection_info() -> str:
    """Return a one-line summary of what's in the Qdrant collection."""
    try:
        client = QdrantClient(path=QDRANT_PATH)
        count  = client.count(collection_name=COLLECTION_NAME).count
        return f"[success]✔  Collection '{COLLECTION_NAME}':[/success] [bold]{count}[/bold] chunks indexed"
    except Exception:
        return f"[warning]⚠  Collection '{COLLECTION_NAME}' not found — run ingestion first[/warning]"


def run_interactive():
    """Interactive REPL loop."""
    console.print(Panel(
        "[bold]PDF RAG — Multi-Agent Q&A System[/bold]\n"
        "[dim]Powered by:[/dim]  Gemini AI · Qdrant · LangGraph · BM25 · Cross-Encoder\n\n"
        + _collection_info() + "\n\n"
        "[dim]Type your question and press Enter.\n"
        "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.[/dim]",
        title="🤖 RAG Q&A",
        border_style="blue",
    ))

    while True:
        console.print()
        try:
            question = console.input("[bold cyan]Your question:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        # ── Run the full pipeline ─────────────────────────────────────────
        console.print()
        answer = ask(question)

        # ── Print the final answer ────────────────────────────────────────
        console.print()
        console.print(Panel(
            answer,
            title="📋 ANSWER",
            border_style="green",
            padding=(1, 2),
        ))
        console.print(Rule(style="dim"))


def run_single(question: str):
    """Non-interactive mode: answer one question and exit."""
    console.print(Panel(
        "[bold]PDF RAG — Multi-Agent Q&A System[/bold]\n" + _collection_info(),
        title="🤖 RAG Q&A",
        border_style="blue",
    ))
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
    answer = ask(question)
    console.print(Panel(answer, title="📋 ANSWER", border_style="green", padding=(1, 2)))


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about your indexed PDF documents."
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help="Single question (non-interactive mode)",
    )
    args = parser.parse_args()

    if args.question:
        run_single(args.question)
    else:
        run_interactive()


if __name__ == "__main__":
    main()

