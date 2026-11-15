"""
sanitizer.py — Strip Embedded Injections Before Indexing
----------------------------------------------------------
Scans each chunk for text patterns that look like prompt-injection
attempts hidden inside PDFs (e.g., "ignore your instructions...").

This is Layer 1 of the security model — we clean the data *before*
it ever reaches the vector database.
"""

import re
from src.config import console

# Common injection markers hidden inside documents
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+)?instructions",
    r"you are now",
    r"new\s+persona",
    r"disregard\s+your",
    r"forget\s+everything",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+everything\s+above",
    r"repeat\s+(the\s+)?instructions",
    r"act\s+as\s+(a\s+|an\s+)?(different|new|evil|unrestricted)",
    r"jailbreak",
    r"DAN\s+mode",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def sanitize_chunk(text: str) -> tuple[str, bool]:
    """
    Scan one chunk for injection patterns.

    Returns:
        (cleaned_text, was_modified)
        Suspicious phrases are replaced with [REDACTED].
    """
    modified = False
    for pattern in _COMPILED:
        if pattern.search(text):
            text = pattern.sub("[REDACTED]", text)
            modified = True
    return text, modified


def sanitize_chunks(chunks: list[dict]) -> list[dict]:
    """
    Sanitize all chunks in-place.
    Logs how many were flagged.
    """
    flagged = 0
    for chunk in chunks:
        cleaned, was_modified = sanitize_chunk(chunk["text"])
        chunk["text"] = cleaned
        if was_modified:
            flagged += 1

    if flagged:
        console.print(
            f"  [warning]⚠  {flagged} chunk(s) contained suspicious patterns — "
            "phrases redacted.[/warning]"
        )
    else:
        console.print(
            f"  [success]✔[/success]  {len(chunks)} chunks scanned — no injection patterns found"
        )
    return chunks

