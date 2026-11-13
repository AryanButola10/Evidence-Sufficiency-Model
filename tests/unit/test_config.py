"""
tests/unit/test_config.py

Unit tests for the configuration system (Phase 1).

These tests verify that:
    - Settings load correctly from environment
    - ESM threshold validation works
    - LLM provider configuration validates correctly

Run with:
    pytest tests/unit/test_config.py -v
"""
from __future__ import annotations

import os
import pytest

from config.settings import Settings


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_llm_provider(self):
        """Default LLM provider should be ollama."""
        s = Settings()
        assert s.llm_provider == "ollama"

    def test_default_esm_thresholds(self):
        """Default ESM thresholds should be 0.35 (lower) and 0.65 (upper)."""
        s = Settings()
        assert s.esm_lower_threshold == 0.35
        assert s.esm_upper_threshold == 0.65

    def test_esm_threshold_order_validation(self):
        """validate_llm_config should raise if lower >= upper."""
        s = Settings(esm_lower_threshold=0.7, esm_upper_threshold=0.5)
        with pytest.raises(ValueError, match="ESM_LOWER_THRESHOLD"):
            s.validate_llm_config()

    def test_groq_validation_raises_without_key(self, monkeypatch):
        """Groq provider should raise ValueError if API key is missing."""
        s = Settings(llm_provider="groq", groq_api_key="")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            s.validate_llm_config()

    def test_groq_validation_passes_with_key(self):
        """Groq provider should pass validation when API key is present."""
        s = Settings(llm_provider="groq", groq_api_key="test_key_12345")
        # Should not raise
        s.validate_llm_config()

    def test_cors_origins_parsed_from_string(self):
        """CORS origins should be parseable from comma-separated string."""
        s = Settings(cors_origins="http://localhost:3000,http://localhost:3001")
        assert "http://localhost:3000" in s.cors_origins
        assert "http://localhost:3001" in s.cors_origins

    def test_project_root_property(self):
        """Project root should point to the project directory."""
        s = Settings()
        assert s.project_root.exists()
        assert (s.project_root / "config").exists()

    def test_max_retrieval_iterations_range(self):
        """Max retrieval iterations should accept valid values."""
        s = Settings(esm_max_retrieval_iterations=5)
        assert s.esm_max_retrieval_iterations == 5

    def test_embedding_device_default(self):
        """Default embedding device should be mps for Apple Silicon."""
        s = Settings()
        assert s.embedding_device == "mps"
