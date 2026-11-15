"""
chunker.py — Pages → Chunks
-----------------------------
Splits each page's text into smaller, overlapping segments using
LangChain's RecursiveCharacterTextSplitter.

Why overlap?
  If a sentence is split across two chunks, the overlap keeps enough
  context in each chunk so retrieval still finds the right answer.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import console, CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(
    pages: list[dict],
    chunk_size: int  = CHUNK_SIZE,
    overlap: int     = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split page texts into overlapping chunks.

    Returns:
        Flat list of chunk dicts:
        {
            "text": str,
            "metadata": { ...page metadata..., "chunk_index": int }
        }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size     = chunk_size,
        chunk_overlap  = overlap,
        length_function= len,
        separators     = ["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        texts = splitter.split_text(page["text"])
        for i, text in enumerate(texts):
            text = text.strip()
            if text:
                chunks.append({
                    "text"    : text,
                    "metadata": {**page["metadata"], "chunk_index": i},
                })

    console.print(
        f"  [success]✔[/success]  Created [bold]{len(chunks)}[/bold] chunks  "
        f"[dim](size={chunk_size}, overlap={overlap})[/dim]"
    )
    return chunks
