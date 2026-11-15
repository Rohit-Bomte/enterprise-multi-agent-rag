# Mini Enterprise AI Knowledge Assistant
### Cloud-Native Multi-Agent RAG with Hybrid Search, Security Guardrails & Kubernetes Deployment

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-dc2626?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-4285F4?logo=google&logoColor=white)](https://aistudio.google.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Cloud--Native-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)

---

## 📌 Executive Summary

**Mini Enterprise AI Knowledge Assistant** is an end-to-end, production-grade Retrieval-Augmented Generation (RAG) platform. It enables enterprise users to upload dense technical PDF documents, generate high-dimensional embeddings, and execute grounded, citation-backed question answering through an autonomous **3-agent workflow**.

The platform is designed with a **defense-in-depth security framework** (aligned with the OWASP LLM Top 10), hybrid semantic and lexical retrieval (Dense Vectors + BM25 + Reciprocal Rank Fusion + Local Cross-Encoder Reranker), cloud-native containerization, and centralized Prometheus metrics and structured JSON logging.

---

## 🏛️ End-to-End Architecture

```
                          ┌────────────────────────────────────────────────────────┐
                          │               OFFLINE INGESTION PIPELINE               │
                          │                                                        │
  ┌──────────────┐        │ ┌──────────────┐   sanitize  ┌──────────────┐          │
  │ Uploaded PDF │───────▶│ │ PyMuPDF      │────────────▶│ Injection    │          │
  │ Documents    │        │ │ Page Parser  │             │ Sanitizer    │          │
  └──────────────┘        │ └──────────────┘             └──────┬───────┘          │
                          │                                     │                  │
                          │                                     ▼                  │
                          │ ┌──────────────┐    embed    ┌──────────────┐          │
                          │ │ Qdrant Vector│◀────────────│ Recursive    │          │
                          │ │ & BM25 Store │ (3072-dim)  │ Chunker      │          │
                          │ └──────────────┘             └──────────────┘          │
                          └────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                          ┌────────────────────────────────────────────────────────┐
                          │             ONLINE MULTI-AGENT QUERY PIPELINE          │
                          │                                                        │
  ┌──────────────┐        │ ┌───────────────────────────┐                          │
  │ User Prompt  │───────▶│ │ Layer 2: Input Guardrail  │ (Regex firewall & length)│
  └──────────────┘        │ └─────────────┬─────────────┘                          │
                          │               │ (Clean Query)                          │
                          │               ▼                                        │
                          │ ┌───────────────────────────┐                          │
                          │ │ Agent 1: Research Agent   │ (Intent & Query Rewrite) │
                          │ └─────────────┬─────────────┘                          │
                          │               │ (Optimized Query)                      │
                          │               ▼                                        │
                          │ ┌───────────────────────────┐                          │
                          │ │ Agent 2: Retrieval Agent  │ (Dense + BM25 + RRF +    │
                          │ └─────────────┬─────────────┘  Cross-Encoder Reranker) │
                          │               │ (Top-5 Re-ranked Chunks)               │
                          │               ▼                                        │
                          │ ┌───────────────────────────┐                          │
                          │ │ Agent 3: Generation Agent │ (Grounded in XML bounds) │
                          │ └─────────────┬─────────────┘                          │
                          │               │ (Cited Response)                       │
                          │               ▼                                        │
                          │ ┌───────────────────────────┐                          │
                          │ │ Layer 4: Output Guardrail │ (PII mask & leak block)  │
  ┌──────────────┐        │ └─────────────┬─────────────┘                          │
  │ Verified     │◀───────────────────────┘                                        │
  │ Cited Answer │        │                                                        │
  └──────────────┘        └────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Differentiators & Features

1. **3-Agent LangGraph StateGraph**:
   - **Research Agent**: Decodes ambiguous questions and optimizes keyword/semantic queries.
   - **Data Retrieval Agent**: Executes multi-stage retrieval and re-ranks candidate passages.
   - **Response Generation Agent**: Synthesizes verified answers strictly citing `[Source: document.pdf, Page N]`.
2. **Hybrid Search with Reciprocal Rank Fusion (RRF)**:
   - Combines dense embeddings (`gemini-embedding-001`, 3072 dimensions) and lexical sparse search (`rank-bm25`).
   - Re-ranks candidate passages using a local cross-encoder (`ms-marco-MiniLM-L-6-v2`) to eliminate false positives.
3. **4-Layer Defense-in-Depth Security**:
   - **Layer 1 (Ingestion)**: Scans and neutralizes adversarial instructions embedded inside PDF text.
   - **Layer 2 (Input Firewall)**: Blocks direct prompt injection, DAN jailbreaks, and length-bomb attacks.
   - **Layer 3 (Prompt Boundary)**: Enforces strict XML delimitation (`<retrieved_context>`) to treat retrieved data as inert.
   - **Layer 4 (Output Sanitizer)**: Automatically masks PII (SSNs, emails, phone numbers) and blocks system prompt leakage.
4. **Cloud-Native & Enterprise Observability**:
   - **Prometheus Metrics**: Pre-instrumented `/metrics` tracking query count, agent latency distributions, and guardrail interception rates.
   - **Structured JSON Logging**: Standardized JSON logs ready for Loki, FluentBit, or Elasticsearch ingestion.
   - **Production Readiness Probes**: `/healthz` endpoint for Kubernetes liveness and readiness checks.

---

## 📁 Repository Structure

```
.
├── .dockerignore                         # Docker build exclusions
├── .gitignore                            # Standard git ignore rules
├── .env.example                          # Environment template
├── Dockerfile                            # Production multi-stage Docker build with pre-cached models
├── docker-compose.yml                    # Local multi-service orchestration (App + Qdrant + Prometheus)
├── requirements.txt                      # Pinned production Python dependencies
├── README.md                             # Enterprise documentation
│
├── data/                                 # PDF ingestion directory
│   └── .gitkeep
│
├── src/                                  # Application Source Code
│   ├── __init__.py
│   ├── config.py                         # Centralized configuration & dynamic Qdrant connector
│   ├── api.py                            # Production FastAPI REST microservice
│   ├── main.py                           # CLI interactive chat & script runner
│   ├── workflow.py                       # LangGraph multi-agent state machine
│   │
│   ├── agents/                           # Autonomous Specialist Agents
│   │   ├── __init__.py
│   │   ├── research_agent.py             # Agent 1: Query optimizer
│   │   ├── retrieval_agent.py            # Agent 2: Hybrid search & reranker
│   │   └── generation_agent.py           # Agent 3: Grounded synthesizer
│   │
│   ├── ingestion/                        # Document Processing Pipeline
│   │   ├── __init__.py
│   │   ├── parser.py                     # Page-level PDF text extractor (PyMuPDF)
│   │   ├── sanitizer.py                  # Document injection sanitizer (Layer 1)
│   │   ├── chunker.py                    # Recursive overlapping chunker
│   │   └── indexer.py                    # Vector DB & BM25 indexer
│   │
│   ├── retrieval/                        # Retrieval Engine
│   │   ├── __init__.py
│   │   ├── embedder.py                   # Gemini query embedding service
│   │   ├── bm25_retriever.py             # Lexical keyword search (BM25Okapi)
│   │   └── hybrid_retriever.py           # RRF fusion & local Cross-Encoder reranking
│   │
│   ├── guardrails/                       # Security Guardrails
│   │   ├── __init__.py
│   │   ├── input_guard.py                # Pre-execution prompt injection filter (Layer 2)
│   │   └── output_guard.py               # Post-execution PII & leakage redactor (Layer 4)
│   │
│   └── observability/                    # Centralized Logging & Metrics
│       ├── __init__.py
│       ├── logger.py                     # Structured JSON logging
│       └── metrics.py                    # Prometheus metrics & custom histograms
│
├── static/                               # Modern Web Interface
│   └── index.html                        # Drag-and-drop upload & interactive chat UI
│
├── k8s/                                  # Production Kubernetes Manifests
│   ├── 00-namespace.yaml                 # Dedicated 'enterprise-rag' namespace
│   ├── 01-secrets-configmap.yaml         # App config and API key secrets
│   ├── 02-qdrant-statefulset.yaml        # Qdrant StatefulSet with PersistentVolumeClaim
│   ├── 03-qdrant-service.yaml            # Internal ClusterIP service for vector DB
│   ├── 04-rag-app-deployment.yaml        # Autoscaling RAG API pods with health probes
│   ├── 05-rag-app-service.yaml           # Service exposing API port 8000
│   ├── 06-ingress.yaml                   # Ingress controller routing rule
│   └── 07-monitoring.yaml                # In-cluster Prometheus deployment
│
├── monitoring/                           # Observability Configuration
│   └── prometheus.yml                    # Prometheus scrape configuration
│
└── tests/                                # Test Automation Suite
    ├── __init__.py
    ├── test_adversarial.py               # 16 OWASP adversarial unit tests
    └── test_api.py                       # 5 FastAPI integration & guardrail tests
```

---

## ⚡ Quickstart: Local Environment Setup

### 1. Prerequisites
- Python 3.11+
- Free Google Gemini API Key: [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/enterprise-rag-assistant.git
cd enterprise-rag-assistant

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows
# source .venv/bin/activate     # On Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Credentials
```powershell
copy .env.example .env
# Edit .env and supply your GEMINI_API_KEY
```

### 4. Ingest Sample Document
```powershell
python -m src.ingest_pipeline "data/your_document.pdf"
```

### 5. Launch the Local Web App
```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to access the interactive web interface, or **[http://localhost:8000/docs](http://localhost:8000/docs)** for the Swagger REST API documentation.

---

## 🐳 Containerized Deployment (Docker Desktop)

The included `docker-compose.yml` deploys the full stack:
- **`rag-api`**: The Multi-Agent RAG FastAPI microservice (port 8000)
- **`qdrant`**: Dedicated Vector Database with persistent storage (ports 6333, 6334)
- **`prometheus`**: Observability agent scraping `/metrics` (port 9090)

```powershell
# Build and run all services
docker compose up -d --build

# View real-time structured logs
docker compose logs -f rag-api

# Check service status
docker compose ps
```

Access points:
- **Web UI & API**: [http://localhost:8000](http://localhost:8000)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **Prometheus Console**: [http://localhost:9090](http://localhost:9090)

---

## ☸️ Production Kubernetes Deployment

Deploy to any Kubernetes cluster (Docker Desktop Kubernetes, Minikube, EKS, GKE, AKS):

### 1. Configure Secrets
Edit `k8s/01-secrets-configmap.yaml` and set your `GEMINI_API_KEY`:
```yaml
stringData:
  GEMINI_API_KEY: "AIzaSy..."
```

### 2. Apply Manifests
```powershell
# Create namespace, configs, vector DB, app deployments, and services
kubectl apply -f k8s/
```

### 3. Verify Deployment
```powershell
kubectl get pods -n enterprise-rag -w
kubectl get services -n enterprise-rag
```

### 4. Port Forward to Access Locally
```powershell
kubectl port-forward svc/rag-api-service 8000:8000 -n enterprise-rag
```

---

## 📊 Centralized Observability & Metrics

Prometheus collects operational metrics from `/metrics`:

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `rag_queries_total` | Counter | Total queries processed, labeled by status (`success`, `refused`, `error`) |
| `rag_query_duration_seconds` | Histogram | Latency distribution of end-to-end multi-agent execution |
| `rag_guardrail_blocks_total` | Counter | Intercepted threats labeled by layer (`input`, `output`) and reason |
| `rag_chunks_indexed_total` | Counter | Total chunks indexed, labeled by source file |
| `http_requests_total` | Counter | Standard HTTP status code distributions |

---

## 🛡️ Security & Adversarial Test Suite

The system includes an automated test suite verifying guardrail efficacy:
- **Prompt Injection Defense**: Evaluates resistance to "ignore previous instructions" overrides.
- **Jailbreak Defense**: Evaluates resistance to DAN persona shifts.
- **PII Leakage Prevention**: Validates redaction of SSNs, emails, and phone numbers in generated output.
- **System Prompt Integrity**: Confirms model instructions are not disclosed.

```powershell
# Run the entire test suite
pytest tests/ -v
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
