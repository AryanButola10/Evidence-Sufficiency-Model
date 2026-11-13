# Reliability-Oriented RAG Framework

**A Reliability-Oriented RAG Framework Using Evidence Sufficiency Modeling**

B.Tech CSE (AI & DS) — Major Research Project  
Graphic Era Deemed to be University, Dehradun

---

## Research Overview

This project proposes a RAG framework that explicitly evaluates the **sufficiency and reliability** of retrieved evidence before generating an answer.

The system selects one of three evidence-driven actions:
- **ANSWER** — when evidence is sufficient and reliable
- **RETRIEVE MORE** — when evidence is incomplete but further retrieval may help
- **ABSTAIN** — when reliable evidence cannot be established

The central contribution is an **Evidence Sufficiency Model (ESM)** that jointly evaluates five evidence characteristics: relevance, entailment, contradiction, source reliability, and coverage.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11 + FastAPI |
| LLM (local) | Ollama + Llama 3.2/3.1 |
| LLM (API) | Groq API |
| Semantic retrieval | BGE-M3 + Qdrant |
| Lexical retrieval | BM25 (rank-bm25) |
| Reranker | BAAI/bge-reranker-base |
| NLI (evidence evaluation) | DeBERTa-v3 |
| Frontend | Next.js (App Router) + TypeScript |
| Experiment tracking | MLflow |

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Complete | Architecture and planning |
| Phase 1 | 🔄 In Progress | Environment setup |
| Phase 2 | ⏳ Pending | Document processing |
| Phase 3 | ⏳ Pending | Hybrid retrieval |
| Phase 4 | ⏳ Pending | Evidence evaluation |
| Phase 5 | ⏳ Pending | Evidence Sufficiency Model |
| Phase 6 | ⏳ Pending | Adaptive decision module |
| Phase 7 | ⏳ Pending | LLM integration + answer generation |
| Phase 8 | ⏳ Pending | Claim verification + provenance |
| Phase 9 | ⏳ Pending | Frontend + API integration |
| Phase 10 | ⏳ Pending | Experimental evaluation |
| Phase 11 | ⏳ Pending | Finalization |

---

## Setup Instructions

### Prerequisites

- macOS (Apple Silicon M4)
- Python 3.11 (`brew install python@3.11`)
- Ollama (`https://ollama.ai/download`)
- Node.js 18+ (for frontend)

### Environment Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd reliability-rag

# 2. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment with Python 3.11
uv venv .venv --python 3.11
source .venv/bin/activate

# 4. Install Phase 1 dependencies
uv pip install -r requirements.txt

# 5. Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# 6. Pull Ollama models
ollama pull llama3.2:3b    # Dev model (~2 GB)

# 7. Start the API server
uvicorn api.main:app --reload

# 8. Run the health check
python scripts/health_check.py

# 9. Run tests
pytest tests/unit/ -v
```

### Verifying the Setup

After starting the server, visit:
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## Datasets

| Dataset | Role | Purpose |
|---------|------|---------|
| **HotpotQA** | Primary | Multi-hop QA, evidence coverage, retrieval evaluation |
| **RAGTruth** | Secondary | Hallucination detection, claim verification, faithfulness |

---

## Repository Structure

```
reliability-rag/
├── config/           # Settings and logging
├── src/
│   ├── document_processing/
│   ├── retrieval/
│   ├── evidence_evaluation/
│   ├── evidence_sufficiency/
│   ├── decision/
│   ├── generation/   # LLM provider abstraction
│   ├── verification/
│   ├── orchestration/
│   └── evaluation/
├── api/              # FastAPI application
├── frontend/         # Next.js (Phase 9)
├── tests/
├── notebooks/
├── experiments/
└── docs/
```
