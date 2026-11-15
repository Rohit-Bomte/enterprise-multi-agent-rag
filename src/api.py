"""
api.py — Production Enterprise FastAPI Service
------------------------------------------------
Provides cloud-native REST endpoints for the PDF Multi-Agent RAG system:
  - POST /api/v1/query   : Ask questions across ingested documents
  - POST /api/v1/upload  : Upload and automatically index PDF files
  - GET  /api/v1/stats   : Vector DB status & collection metrics
  - GET  /healthz        : Kubernetes Liveness & Readiness probe
  - GET  /metrics        : Prometheus metrics scraper endpoint
  - GET  /               : Embedded enterprise web interface
"""

import os
import time
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    DATA_DIR, COLLECTION_NAME, get_qdrant_client,
    GEMINI_LLM_MODEL, GEMINI_EMBED_MODEL
)
from src.workflow import ask, app as workflow_app
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.sanitizer import sanitize_chunks
from src.ingestion.indexer import index_chunks
from src.observability.logger import get_logger, setup_logging
from src.observability.metrics import (
    init_metrics,
    rag_query_counter,
    rag_query_latency,
    rag_chunks_indexed,
    rag_guardrail_blocks,
)

# Initialize production logging
setup_logging()
logger = get_logger("enterprise_rag.api")


# ── Lifespan Context Manager ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Enterprise RAG Microservice...", extra={"event": "startup"})
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        logger.info(
            f"Connected to Qdrant successfully. Collections: {collections}",
            extra={"event": "qdrant_connected", "collections": collections}
        )
    except Exception as e:
        logger.warning(
            f"Qdrant connection deferred or not ready: {e}",
            extra={"event": "qdrant_warn", "error": str(e)}
        )
    yield
    logger.info("Shutting down Enterprise RAG Microservice...", extra={"event": "shutdown"})


app = FastAPI(
    title="Mini Enterprise AI Knowledge Assistant",
    description="Multi-Agent RAG with Hybrid Search, Cross-Encoder Reranking, and 4-Layer Security Guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)

# Expose /metrics for Prometheus scraping
init_metrics(app)

# Mount static frontend
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic Request / Response Schemas ───────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User's natural language question")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the operating frequency of the mmWave human detection radar?"
            }
        }
    }


class QueryResponse(BaseModel):
    question: str
    answer: str
    is_safe: bool
    latency_seconds: float


class IngestResponse(BaseModel):
    filename: str
    pages: int
    chunks_indexed: int
    status: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def root():
    """Serves the modern enterprise interactive UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Enterprise AI Knowledge Assistant</h1>"
        "<p>API running. Visit <a href='/docs'>/docs</a> for Swagger UI.</p>"
    )


@app.get("/healthz", tags=["Kubernetes Probes"])
def health_check():
    """
    Kubernetes Liveness & Readiness Probe.
    Returns 200 OK if service is healthy and Qdrant is reachable.
    """
    try:
        client = get_qdrant_client()
        cols = [c.name for c in client.get_collections().collections]
        count = 0
        if COLLECTION_NAME in cols:
            count = client.count(collection_name=COLLECTION_NAME).count
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "qdrant": "connected",
                "collection": COLLECTION_NAME,
                "collection_exists": COLLECTION_NAME in cols,
                "total_chunks_indexed": count,
                "llm_model": GEMINI_LLM_MODEL,
                "embed_model": GEMINI_EMBED_MODEL,
            }
        )
    except Exception as e:
        logger.error(f"Health check failure: {e}", extra={"event": "healthz_error"})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Workflow"])
def query_endpoint(req: QueryRequest):
    """
    Executes the 3-Agent RAG Workflow:
      Input Guardrail → Research Agent → Data Retrieval Agent → Generation Agent → Output Guardrail
    """
    start_time = time.time()
    logger.info(f"Incoming query: {req.question[:80]}...", extra={"event": "query_start"})

    try:
        initial_state = {
            "original_question": req.question,
            "is_safe": True,
            "rewritten_query": "",
            "reranked_chunks": [],
            "final_answer": "",
        }

        # Invoke LangGraph StateGraph
        result = workflow_app.invoke(initial_state)

        latency = round(time.time() - start_time, 3)
        is_safe = result.get("is_safe", True)
        answer = result.get("final_answer", "")

        # Record metrics
        status_label = "success" if is_safe else "refused"
        rag_query_counter.labels(status=status_label).inc()
        rag_query_latency.observe(latency)

        if not is_safe:
            rag_guardrail_blocks.labels(layer="pipeline", reason="policy_violation").inc()

        logger.info(
            f"Query completed in {latency}s (is_safe={is_safe})",
            extra={"event": "query_complete", "latency_s": latency, "is_safe": is_safe}
        )

        return QueryResponse(
            question=req.question,
            answer=answer,
            is_safe=is_safe,
            latency_seconds=latency,
        )

    except Exception as e:
        rag_query_counter.labels(status="error").inc()
        logger.exception(f"Internal error processing query: {e}", extra={"event": "query_error"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating response: {str(e)}"
        )


@app.post("/api/v1/upload", response_model=IngestResponse, tags=["Document Ingestion"])
def upload_document(
    file: UploadFile = File(...),
    reset: bool = False
):
    """
    Upload and index a PDF file:
      1. Save uploaded file into data/
      2. Extract text (PyMuPDF)
      3. Sanitize for embedded prompt injections
      4. Recursive chunking with overlap
      5. Generate Gemini embeddings & store in Qdrant + BM25 corpus
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    saved_path = DATA_DIR / file.filename
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Received file: {file.filename}", extra={"event": "upload_received", "file": file.filename})

        # Run ingestion pipeline
        pages = parse_pdf(str(saved_path))
        temp_chunks = [{"text": p["text"], "metadata": p["metadata"]} for p in pages]
        temp_chunks = sanitize_chunks(temp_chunks)
        for page, chunk in zip(pages, temp_chunks):
            page["text"] = chunk["text"]

        chunks = chunk_pages(pages)
        index_chunks(chunks, reset=reset)

        rag_chunks_indexed.labels(source_file=file.filename).inc(len(chunks))

        return IngestResponse(
            filename=file.filename,
            pages=len(pages),
            chunks_indexed=len(chunks),
            status="indexed",
            message=f"Successfully indexed {len(chunks)} chunks from {len(pages)} pages."
        )

    except Exception as e:
        logger.exception(f"Document ingestion failed: {e}", extra={"event": "ingest_error"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@app.get("/api/v1/stats", tags=["System Information"])
def get_stats():
    """Returns vector store and pipeline metrics."""
    try:
        client = get_qdrant_client()
        count = client.count(collection_name=COLLECTION_NAME).count
        return {
            "collection_name": COLLECTION_NAME,
            "total_points": count,
            "llm_model": GEMINI_LLM_MODEL,
            "embedding_model": GEMINI_EMBED_MODEL,
            "architecture": "Hybrid (Dense + BM25Okapi + RRF + CrossEncoder)",
            "guardrails": ["Input Regex Guard", "Output PII Redactor", "Document Sanitizer"],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
