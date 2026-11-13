# A Reliability-Oriented RAG Framework Using Evidence Sufficiency Modeling

**Student:** B.Tech CSE (AI & DS), 7th Semester — Graphic Era Deemed to be University  
**Project Type:** Major Academic Research Project  
**Hardware:** MacBook Air M4, 16 GB RAM (Apple Silicon)

---

## 1. Project Understanding

### Research Objective (Plain Language)

Standard RAG systems retrieve documents and generate answers — but they never ask a critical question: *"Is this evidence actually good enough to answer reliably?"*

This project builds a system that explicitly evaluates the **quality and sufficiency** of retrieved evidence before deciding what to do. Instead of always generating an answer, the system can:

- **Answer** — when evidence is strong and sufficient
- **Retrieve More** — when evidence is weak but targeted retrieval may help
- **Abstain** — when reliable evidence cannot be established

The central research hypothesis is: *"An explicit Evidence Sufficiency Model that jointly evaluates relevance, entailment, contradiction, source reliability, and coverage can improve answer reliability and decision-making in RAG systems compared to conventional RAG."*

---

## 2. Central Research Contribution

The key novelty is the **Evidence Sufficiency Model (ESM)** — a module that:

1. Takes multiple evidence quality signals as input
2. Jointly aggregates them into a single interpretable sufficiency score
3. Drives an adaptive decision (Answer / Retrieve More / Abstain)

This is **not** just a similarity threshold. It is a principled, multi-dimensional evidence assessment that can be:
- Experimentally ablated (remove one signal at a time)
- Compared against baselines
- Evaluated for calibration quality

---

## 3. Approved System Architecture Summary

```
USER QUERY
    │
    ▼
HYBRID EVIDENCE RETRIEVAL ──────────── BM25 (lexical) + BGE-M3 (semantic)
    │
    ▼
RESULT FUSION + RERANKING ──────────── Reciprocal Rank Fusion + BGE Reranker
    │
    ▼
EVIDENCE EVALUATION ─────────────────── Relevance · Entailment · Contradiction
    │                                    Source Reliability · Coverage
    ▼
EVIDENCE SUFFICIENCY MODEL ─────────── Multi-signal aggregation → ESM Score
    │
    ▼
ADAPTIVE DECISION ───────────────────── ANSWER / RETRIEVE MORE / ABSTAIN
    │
    ▼
ANSWER GENERATION ───────────────────── LLM (Qwen/Llama) with evidence grounding
    │
    ▼
CLAIM-LEVEL VERIFICATION ───────────── NLI-based claim checking
    │
    ▼
PROVENANCE + CITATIONS ─────────────── Claim → Source mapping
    │
    ▼
FINAL RESPONSE (with transparency)
```

---

## 4. Practical Implementation Strategy for MacBook Air M4 (16 GB RAM)

### Hardware Context

| Resource | Available | Notes |
|----------|-----------|-------|
| RAM | 16 GB unified memory | Shared between CPU and GPU (Apple MPS) |
| Chip | M4 (Apple Silicon) | Supports MPS acceleration, NOT CUDA |
| Storage | Depends on config | Models + Qdrant + PostgreSQL |
| Internet | Assumed available | For API fallbacks |

### Strategy: Local-First, API-Fallback

| Component | Local or API? | Reasoning |
|-----------|--------------|-----------|
| BGE-M3 embeddings | ✅ Local (MPS) | ~1.1 GB model, runs fine on M4 |
| BM25 retrieval | ✅ Local | Pure Python, no GPU needed |
| BGE Reranker (small) | ✅ Local (MPS) | ~0.5 GB, feasible |
| DeBERTa-v3 (NLI) | ✅ Local (MPS) | ~0.7 GB, feasible |
| Qdrant | ✅ Local (Docker or in-memory) | Lightweight |
| PostgreSQL | ✅ Local | Standard DB |
| LLM for generation | ⚠️ Hybrid | Qwen2.5-7B quantized (~5-6 GB) OR Groq/Together API |
| Claim extraction LLM | ⚠️ Hybrid | Same as above — same model reused |

> **Recommended LLM approach:** Start with **Groq API** (free tier, very fast) using `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`. Once the pipeline is proven, optionally switch to a local quantized model like `Qwen2.5-7B-Q4_K_M` via `llama.cpp` or `ollama`.

### Critical Memory Budgets (16 GB)

```
BGE-M3 embeddings model:   ~1.1 GB
BGE Reranker (small):      ~0.5 GB
DeBERTa-v3-large (NLI):    ~0.9 GB
Qdrant in-memory:          ~0.5-2 GB (depends on corpus size)
OS + other processes:      ~4-5 GB
Remaining for LLM:         ~7-9 GB
Qwen2.5-7B Q4 (local):     ~5-6 GB ✅ fits
```

If running all models locally simultaneously is tight, we will pipeline them (load/unload as needed) or offload LLM to API.

---

## 5. Modular Repository Structure

```
reliability-rag/
│
├── README.md
├── .env.example                    # Template for secrets
├── .gitignore
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Qdrant + PostgreSQL
├── pyproject.toml                  # Project metadata
│
├── config/
│   ├── settings.py                 # Pydantic settings (loads .env)
│   └── logging_config.py          # Logging setup
│
├── data/
│   ├── raw/                        # Uploaded PDFs
│   ├── processed/                  # Extracted chunks (JSON/JSONL)
│   └── datasets/                   # Evaluation datasets
│
├── src/
│   ├── __init__.py
│   │
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py        # PyMuPDF text extraction
│   │   ├── text_cleaner.py         # Text normalization
│   │   ├── chunker.py              # Text chunking strategies
│   │   └── metadata_manager.py     # Chunk metadata preservation
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py       # BM25 lexical retrieval
│   │   ├── semantic_retriever.py   # BGE-M3 + Qdrant
│   │   ├── fusion.py               # Reciprocal Rank Fusion
│   │   ├── reranker.py             # BGE Reranker
│   │   └── hybrid_retriever.py     # Orchestrates the full pipeline
│   │
│   ├── evidence_evaluation/
│   │   ├── __init__.py
│   │   ├── relevance_scorer.py     # Semantic relevance scoring
│   │   ├── entailment_scorer.py    # NLI-based entailment
│   │   ├── contradiction_detector.py # NLI-based contradiction
│   │   ├── source_reliability.py   # Heuristic source trust score
│   │   ├── coverage_estimator.py   # Query aspect coverage
│   │   └── evidence_evaluator.py   # Orchestrates all 5 dimensions
│   │
│   ├── evidence_sufficiency/
│   │   ├── __init__.py
│   │   ├── esm_model.py            # Evidence Sufficiency Model (baseline + variants)
│   │   ├── aggregation.py          # Scoring aggregation strategies
│   │   └── esm_config.py           # Weights, thresholds (configurable)
│   │
│   ├── decision/
│   │   ├── __init__.py
│   │   └── adaptive_decision.py    # ANSWER / RETRIEVE MORE / ABSTAIN logic
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py       # Evidence-grounded prompt construction
│   │   └── answer_generator.py     # LLM interface (local or API)
│   │
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── claim_extractor.py      # Extract factual claims from answer
│   │   ├── claim_verifier.py       # NLI-based claim verification
│   │   └── provenance_mapper.py    # Map claims → source passages
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── pipeline.py             # LangGraph workflow orchestration
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # Answer accuracy, F1, faithfulness, etc.
│       ├── calibration.py          # ECE, Brier Score
│       ├── ablation.py             # Ablation study runner
│       └── baselines.py            # Conventional RAG, hybrid RAG baselines
│
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app entry point
│   ├── routers/
│   │   ├── documents.py            # Upload + process endpoints
│   │   ├── query.py                # Query + answer endpoint
│   │   └── evaluation.py           # Evaluation run endpoint
│   ├── schemas/
│   │   ├── request_schemas.py
│   │   └── response_schemas.py
│   └── dependencies.py             # Dependency injection
│
├── frontend/
│   ├── (Next.js or React application)
│   └── ...
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_fusion.py
│   │   ├── test_esm.py
│   │   └── ...
│   └── integration/
│       └── test_pipeline.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_evaluation.ipynb
│   ├── 03_esm_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── experiments/
│   ├── configs/                    # MLflow experiment configs
│   └── results/                    # Saved experiment results
│
└── docs/
    ├── architecture.md
    ├── research_notes.md
    └── api_reference.md
```

---

## 6. Implementation Phases and Roadmap

| Phase | Title | Key Deliverables | Estimated Effort |
|-------|-------|-----------------|-----------------|
| **0** | Planning | Architecture, repo structure, roadmap | Done (this document) |
| **1** | Environment Setup | Virtual env, dependencies, Git, Docker | 1-2 sessions |
| **2** | Document Processing | PDF extraction, chunking, metadata | 2-3 sessions |
| **3** | Hybrid Retrieval | BM25 + BGE-M3 + Qdrant + RRF + Reranker | 3-4 sessions |
| **4** | Evidence Evaluation | Relevance, entailment, contradiction, reliability, coverage | 4-5 sessions |
| **5** | Evidence Sufficiency Model | ESM baseline, weighted scoring, configurability | 3-4 sessions |
| **6** | Adaptive Decision | Answer/Retrieve More/Abstain logic | 2-3 sessions |
| **7** | Answer Generation | LLM integration, evidence-grounded prompts | 2-3 sessions |
| **8** | Claim Verification | Claim extraction, NLI verification, provenance | 3-4 sessions |
| **9** | Frontend + API Integration | FastAPI + Next.js UI | 4-5 sessions |
| **10** | Experimental Evaluation | Baselines, ablations, metrics, MLflow | 5-6 sessions |
| **11** | Finalization | Docs, demo, report | 3-4 sessions |

> **Total estimated effort:** ~35-50 focused coding sessions  
> **Suggested pace:** 2-3 sessions per week → ~4-6 months

---

## 7. Priority Order for Implementation

The critical path runs through these components in order:

```
Chunker → BM25 → BGE-M3 → Fusion → Reranker → Relevance → Entailment → ESM → Decision → LLM → Verification
```

Phases 2, 3, 4, and 5 are the **research core** — they must be built carefully and tested individually. Do not rush them.

---

## 8. Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| LLM too large for 16 GB RAM | 🔴 High | Use Groq API (free) or Ollama with quantized models |
| BGE-M3 + NLI + Reranker all loaded simultaneously → OOM | 🟡 Medium | Sequential loading; unload between pipeline stages |
| NLI model unreliable for open-domain queries | 🟡 Medium | Use cross-encoder NLI, test on held-out samples |
| Evidence sufficiency labels don't exist in standard datasets | 🟡 Medium | Design a proxy evaluation protocol; document clearly |
| Qdrant + PostgreSQL setup complexity on macOS | 🟠 Low-Medium | Use Docker Compose; fallback to in-memory Qdrant for dev |
| CUDA-specific libraries on Apple Silicon | 🔴 High | Verify all libraries support MPS or CPU; avoid CUDA-only code |
| ESM weights chosen arbitrarily without validation | 🟡 Medium | Treat initial weights as a baseline hypothesis; run ablations |
| LangGraph orchestration adds complexity early | 🟠 Low | Introduce LangGraph in Phase 6+; use simple function calls first |

---

## 9. Local vs. API Component Map

### Runs Fully Locally (Apple MPS / CPU)
- **BM25:** Pure Python (`rank_bm25`) — no GPU needed
- **BGE-M3 embeddings:** `FlagEmbedding` or `sentence-transformers` — MPS supported
- **BGE Reranker (small):** MPS supported, ~0.5 GB
- **DeBERTa-v3-base NLI:** MPS supported, ~0.7 GB
- **Qdrant:** Docker or in-memory mode
- **PostgreSQL:** Docker
- **FastAPI backend:** Pure Python
- **Next.js frontend:** Node.js

### Requires API or Quantized Local Model
- **LLM for answer generation:** Groq API (recommended start) → Ollama local later
- **Claim extraction LLM:** Same as above (reuse)
- **MLflow tracking:** Local server (fully local)

### Optional Cloud Services
- **RAGAS evaluation:** Requires LLM judge — Groq API or OpenAI API
- **Experiment storage:** MLflow local server (no cloud needed)

---

## 10. Experimental Baselines

| Baseline | Description |
|----------|-------------|
| **B1: Vanilla RAG** | BM25 retrieval only + direct LLM generation |
| **B2: Hybrid RAG + Reranking** | BM25 + BGE-M3 + RRF + Reranker + LLM generation |
| **B3: RAG + Evidence Evaluation** | B2 + evidence scoring, but no sufficiency-driven decision |
| **Proposed: Full ESM-RAG** | B2 + Evidence Evaluation + ESM + Adaptive Decision + Claim Verification |

**Ablation experiments** (remove one component at a time from the proposed model):
- Without entailment scoring
- Without contradiction detection
- Without source reliability
- Without coverage estimation
- Without adaptive retrieval (always answer on first retrieval)
- Without abstention (force answer even when ESM is low)

---

## 11. Candidate Datasets

| Dataset | Task | Relevance | Notes |
|---------|------|-----------|-------|
| **Natural Questions (NQ)** | Open-domain QA | ✅ High | Standard QA benchmark |
| **HotpotQA** | Multi-hop QA | ✅ High | Tests multi-document reasoning |
| **TriviaQA** | Trivia QA | ✅ Medium | Tests factual retrieval |
| **RAGTruth** | Hallucination in RAG | ✅ Very High | Directly evaluates RAG faithfulness |
| **FEVER** | Fact verification | ✅ High | Tests contradiction/entailment |
| **FaithDial** | Faithful dialogue | 🟡 Medium | Dialogue-specific |

**Recommended primary datasets:**
1. **RAGTruth** — most directly aligned with the research problem
2. **HotpotQA** — tests multi-document evidence coverage
3. **FEVER** — tests contradiction and entailment handling

> **Important note:** None of these datasets provide "evidence sufficiency" labels directly. We will design a proxy evaluation protocol using answer correctness, faithfulness scores, and NLI-based verification as proxies for sufficiency. This will be explicitly documented in the research methodology.

---

## Open Questions for Your Review

> [!IMPORTANT]
> **LLM Decision:** Do you prefer to start with Groq API (fastest to set up, free tier available) or Ollama with a quantized local model (fully offline, slightly more complex setup)? This affects Phase 1 environment setup.

> [!IMPORTANT]
> **Dataset Priority:** Should we prioritize RAGTruth (hallucination focus) or HotpotQA (multi-hop reasoning) as the primary evaluation dataset? Both are valid — the choice affects experiment design.

> [!NOTE]
> **Frontend Preference:** The approved stack mentions React/Next.js. Do you have a preference for Next.js (full-stack, SSR) or plain React with Vite (simpler, SPA)? For a research demo, Vite + React is often simpler. Next.js is better if you plan to deploy publicly.

> [!NOTE]
> **Qdrant Setup:** For Phase 1, should we start with Qdrant in-memory mode (no Docker needed, simpler) and migrate to Docker-based persistent Qdrant in Phase 3? This keeps Phase 1 lightweight.

---

## Verification Plan for Phase 0

Before proceeding to Phase 1, confirm:
- [ ] Architecture is approved
- [ ] Repository structure is approved  
- [ ] LLM strategy (Groq API vs local) is decided
- [ ] Primary evaluation dataset preference is noted
- [ ] Frontend framework preference is noted
- [ ] You are ready to proceed with environment setup
