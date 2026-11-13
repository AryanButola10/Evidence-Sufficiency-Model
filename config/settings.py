"""
config/settings.py

Central configuration using Pydantic BaseSettings.
All values are loaded from environment variables or .env file.

Usage:
    from config import settings
    print(settings.llm_provider)
    print(settings.ollama_base_url)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root — two levels up from this file
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application-wide settings.

    All fields map directly to environment variables in .env.
    Pydantic automatically coerces types (str → int, str → bool, etc.).
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore unknown env vars
    )

    # -----------------------------------------------------------
    # LLM Provider
    # -----------------------------------------------------------
    llm_provider: Literal["ollama", "groq"] = Field(
        default="ollama",
        description="Which LLM provider to use. 'ollama' = local, 'groq' = API.",
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama server.",
    )
    ollama_dev_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model name for development/testing.",
    )
    ollama_eval_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name for final evaluation runs.",
    )

    # Groq
    groq_api_key: str = Field(
        default="",
        description="Groq API key. Required only when llm_provider=groq.",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model name for API calls.",
    )

    # -----------------------------------------------------------
    # FastAPI
    # -----------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(
        default=True,
        description="Enable auto-reload. Set to false in production.",
    )

    # -----------------------------------------------------------
    # CORS
    # -----------------------------------------------------------
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed origins for CORS. Comma-separated in .env.",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Allow CORS_ORIGINS as a comma-separated string in .env."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # -----------------------------------------------------------
    # Qdrant (Phase 3+)
    # -----------------------------------------------------------
    use_persistent_qdrant: bool = Field(
        default=False,
        description="If false, Qdrant runs in-memory (Phase 1-2 default).",
    )
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="reliability_rag")

    # -----------------------------------------------------------
    # PostgreSQL (Phase 3+)
    # -----------------------------------------------------------
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="reliability_rag")
    postgres_user: str = Field(default="raguser")
    postgres_password: str = Field(default="")

    # -----------------------------------------------------------
    # ML Models (Phase 2+)
    # -----------------------------------------------------------
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_device: Literal["mps", "cpu", "cuda"] = Field(
        default="mps",
        description="Device for embedding model. M4 Mac: use 'mps'.",
    )

    reranker_model: str = Field(default="BAAI/bge-reranker-base")
    reranker_device: Literal["mps", "cpu", "cuda"] = Field(default="mps")

    nli_model: str = Field(default="cross-encoder/nli-deberta-v3-base")
    nli_device: Literal["mps", "cpu", "cuda"] = Field(default="mps")

    # -----------------------------------------------------------
    # Evidence Sufficiency Model Thresholds (Phase 5+)
    # These are initial baseline values — they will be tuned
    # through ablation experiments and are NOT claimed optimal.
    # -----------------------------------------------------------
    esm_upper_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="ESM score above this → ANSWER decision.",
    )
    esm_lower_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="ESM score below this → ABSTAIN decision.",
    )
    esm_max_retrieval_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of RETRIEVE MORE iterations before ABSTAIN.",
    )

    # -----------------------------------------------------------
    # Logging
    # -----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_file: str = Field(default="logs/app.log")

    # -----------------------------------------------------------
    # MLflow (Phase 10+)
    # -----------------------------------------------------------
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="reliability_rag")

    # -----------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"

    @property
    def active_ollama_model(self) -> str:
        """Returns the appropriate Ollama model based on context."""
        return self.ollama_dev_model

    def validate_llm_config(self) -> None:
        """
        Raise a clear error if the LLM configuration is incomplete.
        Called at application startup.
        """
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is missing. "
                "Add GROQ_API_KEY to your .env file."
            )
        if self.esm_lower_threshold >= self.esm_upper_threshold:
            raise ValueError(
                f"ESM_LOWER_THRESHOLD ({self.esm_lower_threshold}) must be "
                f"less than ESM_UPPER_THRESHOLD ({self.esm_upper_threshold})."
            )


# Singleton — import this instance everywhere in the project
settings = Settings()
