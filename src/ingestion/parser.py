"""
parser.py — PDF → Pages
-------------------------
Reads a PDF file page-by-page using PyMuPDF and returns plain text
with metadata attached to each page.
"""

import pymupdf as fitz  # PyMuPDF (new import style)
from pathlib import Path
from src.config import console


def parse_pdf(file_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns:
        list of dicts — one per non-blank page:
        {
            "text": str,
            "metadata": {
                "source": str,       # absolute path
                "filename": str,     # e.g. "report.pdf"
                "page": int,         # 1-based
                "total_pages": int
            }
        }
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    console.print(f"  [dim]Opening:[/dim] [data]{path.name}[/data]")
    doc = fitz.open(str(path))
    total_pages = len(doc)
    pages = []
    scanned_count = 0

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue
        if len(text) < 50:
            scanned_count += 1
        pages.append({
            "text": text,
            "metadata": {
                "source"      : str(path),
                "filename"    : path.name,
                "page"        : page_num + 1,
                "total_pages" : total_pages,
            }
        })

    doc.close()

    if scanned_count > total_pages * 0.3:
        console.print(
            f"  [warning]⚠  {scanned_count}/{total_pages} pages have very little text — "
            "PDF may be scanned. Consider LlamaParse for better results.[/warning]"
        )

    console.print(f"  [success]✔[/success]  Extracted [bold]{len(pages)}[/bold] pages "
                  f"from [data]{path.name}[/data]  ({total_pages} total)")
    return pages
