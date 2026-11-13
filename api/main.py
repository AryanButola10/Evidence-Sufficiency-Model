"""
api/main.py

FastAPI application entry point.

This is the main application file. It:
    1. Creates the FastAPI app instance
    2. Configures CORS (for Next.js frontend on port 3000)
    3. Sets up logging on startup
    4. Validates LLM configuration on startup
    5. Registers all API routers
    6. Provides a /health endpoint for verification

Running the server:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Or with the settings from .env:
    python -m api.main
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_logger, settings
from config.logging_config import setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Code before `yield` runs at startup.
    Code after `yield` runs at shutdown.

    This replaces the deprecated @app.on_event("startup") pattern.
    """
    # ---- STARTUP ----
    setup_logging()
    logger.info("=" * 60)
    logger.info("Reliability-Oriented RAG Framework — Starting Up")
    logger.info("=" * 60)
    logger.info("LLM Provider: {provider}", provider=settings.llm_provider)
    logger.info("Active model:  {model}",
                model=settings.ollama_dev_model if settings.llm_provider == "ollama"
                else settings.groq_model)
    logger.info("API: http://{host}:{port}", host=settings.api_host, port=settings.api_port)

    # Validate LLM config (raises ValueError if misconfigured)
    try:
        settings.validate_llm_config()
        logger.info("LLM configuration validated successfully")
    except ValueError as e:
        logger.error("LLM configuration error: {error}", error=str(e))
        # Don't crash the server — allow it to start for other endpoints
        # The /health endpoint will report the misconfiguration

    logger.info("FastAPI startup complete — ready to accept requests")

    yield  # Application runs here

    # ---- SHUTDOWN ----
    logger.info("Reliability-RAG shutting down — goodbye")


# Create the FastAPI application
app = FastAPI(
    title="Reliability-Oriented RAG Framework",
    description=(
        "A research implementation of an Evidence Sufficiency Model (ESM) "
        "for reliability-oriented retrieval-augmented generation.\n\n"
        "**Research Project:** A Reliability-Oriented RAG Framework Using "
        "Evidence Sufficiency Modeling\n\n"
        "**Institution:** Graphic Era Deemed to be University, Dehradun"
    ),
    version="0.1.0",
    docs_url="/docs",        # Swagger UI at /docs
    redoc_url="/redoc",      # ReDoc at /redoc
    lifespan=lifespan,
)

# ----------------------------------------------------------------
# CORS Middleware
# Allows the Next.js frontend (port 3000) to call this API
# ----------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------
# Health Check Endpoint
# This is the first endpoint to verify everything is working
# ----------------------------------------------------------------
@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Verify the API server is running and report basic configuration.",
)
async def health_check() -> dict:
    """
    Returns the current status of the API server.

    This endpoint is intentionally simple — it checks the server is
    running and configuration is loaded, without hitting any ML models.

    In Phase 7+, this will also check Ollama connectivity and model status.
    """
    return {
        "status": "ok",
        "version": "0.1.0",
        "phase": "Phase 1 — Environment Setup",
        "llm_provider": settings.llm_provider,
        "active_model": (
            settings.ollama_dev_model
            if settings.llm_provider == "ollama"
            else settings.groq_model
        ),
        "qdrant_mode": "in-memory" if not settings.use_persistent_qdrant else "persistent",
        "components_active": {
            "document_processing": False,   # Phase 2
            "hybrid_retrieval": False,      # Phase 3
            "evidence_evaluation": False,   # Phase 4
            "evidence_sufficiency_model": False,  # Phase 5
            "adaptive_decision": False,     # Phase 6
            "answer_generation": False,     # Phase 7
            "claim_verification": False,    # Phase 8
        },
    }


@app.get(
    "/",
    tags=["System"],
    summary="API root",
    include_in_schema=False,
)
async def root() -> dict:
    return {
        "message": "Reliability-Oriented RAG Framework API",
        "docs": "/docs",
        "health": "/health",
    }


# ----------------------------------------------------------------
# Router Registration (Phase 2+)
# Uncomment each router as the corresponding phase is implemented
# ----------------------------------------------------------------
# from api.routers import documents, query, evaluation
# app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
# app.include_router(query.router, prefix="/api/query", tags=["Query"])
# app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])


# ----------------------------------------------------------------
# Direct execution (alternative to uvicorn CLI)
# ----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
