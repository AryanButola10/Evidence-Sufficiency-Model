"""
tests/unit/test_llm_factory.py

Unit tests for the LLM factory and provider abstraction (Phase 1).

These tests verify the factory without actually calling any LLM.
Provider-specific behavior is mocked.

Run with:
    pytest tests/unit/test_llm_factory.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.generation.llm_provider import LLMProvider, LLMResponse
from src.generation.llm_factory import LLMFactory
from src.generation.ollama_provider import OllamaProvider
from src.generation.groq_provider import GroqProvider


class TestLLMResponse:
    """Tests for the LLMResponse data class."""

    def test_response_creation(self):
        """LLMResponse should store all fields correctly."""
        response = LLMResponse(
            content="Test answer",
            model="llama3.2:3b",
            provider="ollama",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert response.content == "Test answer"
        assert response.model == "llama3.2:3b"
        assert response.provider == "ollama"
        assert response.total_tokens_used == 30

    def test_total_tokens_defaults_to_zero(self):
        """total_tokens_used should return 0 when total_tokens is None."""
        response = LLMResponse(
            content="Test",
            model="test-model",
            provider="ollama",
        )
        assert response.total_tokens_used == 0


class TestLLMFactory:
    """Tests for the LLM provider factory."""

    @patch("src.generation.ollama_provider.OllamaProvider.__init__", return_value=None)
    def test_factory_creates_ollama_provider(self, mock_init):
        """Factory should create OllamaProvider when provider='ollama'."""
        result = LLMFactory.create(provider="ollama")
        mock_init.assert_called_once()
        assert isinstance(result, OllamaProvider)

    @patch("src.generation.groq_provider.GroqProvider.__init__", return_value=None)
    @patch("src.generation.llm_factory.settings")
    def test_factory_creates_groq_provider(self, mock_settings, mock_init):
        """Factory should create GroqProvider when provider='groq'."""
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "test_key"
        mock_settings.validate_llm_config = lambda: None
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        result = LLMFactory.create(provider="groq")
        mock_init.assert_called_once()
        assert isinstance(result, GroqProvider)

    def test_factory_raises_for_unknown_provider(self):
        """Factory should raise ValueError for unknown provider names."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create(provider="openai")  # Not supported

    def test_llm_provider_is_abstract(self):
        """LLMProvider should not be directly instantiable."""
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore
