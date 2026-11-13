"""
src/generation/llm_factory.py

LLM Factory — creates the correct provider based on configuration.

This is the single entry point for obtaining an LLM provider instance.
All other modules that need LLM access should use this factory.

Usage:
    from src.generation.llm_factory import LLMFactory

    llm = LLMFactory.create()       # Uses LLM_PROVIDER from .env
    response = llm.generate("...")

Switching providers:
    Set LLM_PROVIDER=ollama  → uses OllamaProvider (local)
    Set LLM_PROVIDER=groq    → uses GroqProvider (API)
    No other code changes needed anywhere.
"""
from __future__ import annotations

from config import get_logger, settings
from src.generation.llm_provider import LLMProvider

logger = get_logger(__name__)


class LLMFactory:
    """
    Factory class for creating LLM provider instances.

    Uses the Strategy pattern — the pipeline code programs against
    the LLMProvider interface, not against a concrete implementation.
    """

    @staticmethod
    def create(provider: str | None = None) -> LLMProvider:
        """
        Create and return the appropriate LLM provider.

        Args:
            provider: Provider name ("ollama" or "groq").
                      If None, reads from settings.llm_provider (.env).

        Returns:
            A concrete LLMProvider instance.

        Raises:
            ValueError: If the provider name is unrecognized.
            RuntimeError: If the provider fails initialization.
        """
        # Import here to avoid circular imports and expensive model loading
        # until the provider is actually needed
        from src.generation.ollama_provider import OllamaProvider
        from src.generation.groq_provider import GroqProvider

        resolved_provider = provider or settings.llm_provider

        logger.info(
            "Creating LLM provider — provider={provider}",
            provider=resolved_provider,
        )

        if resolved_provider == "ollama":
            return OllamaProvider()

        elif resolved_provider == "groq":
            settings.validate_llm_config()  # Ensures GROQ_API_KEY is set
            return GroqProvider()

        else:
            raise ValueError(
                f"Unknown LLM provider: '{resolved_provider}'. "
                f"Valid options: 'ollama', 'groq'. "
                f"Check LLM_PROVIDER in your .env file."
            )

    @staticmethod
    def create_for_evaluation() -> LLMProvider:
        """
        Create the evaluation-quality LLM provider.

        During Phase 10 experiments, this returns a higher-quality
        model (llama3.1:8b for Ollama, llama-3.3-70b for Groq).
        """
        from src.generation.ollama_provider import OllamaProvider
        from src.generation.groq_provider import GroqProvider

        if settings.llm_provider == "ollama":
            # Use the evaluation model (larger, slower, higher quality)
            return OllamaProvider(model=settings.ollama_eval_model)
        else:
            return GroqProvider()  # Groq always uses the eval model
