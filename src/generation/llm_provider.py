"""
src/generation/llm_provider.py

LLM Provider Abstraction Layer.

Design principle:
    The Evidence Sufficiency Model (ESM) and all evidence evaluation modules
    are completely independent of this layer. The LLM is invoked ONLY after
    the ESM has made the ANSWER/RETRIEVE MORE/ABSTAIN decision.

    To switch providers: change LLM_PROVIDER in .env — no code changes needed.

Providers implemented:
    - OllamaProvider: Local LLM via Ollama (primary for development)
    - GroqProvider:   Cloud LLM via Groq API (fallback / final experiments)

Usage:
    from src.generation.llm_factory import LLMFactory
    llm = LLMFactory.create()
    response = llm.generate("What is retrieval-augmented generation?")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    """
    Standardized response object returned by all providers.

    All fields use the same names regardless of which provider generated
    the response. This ensures the rest of the pipeline never needs to
    know which LLM was used.
    """
    content: str                        # The generated text
    model: str                          # Model name that generated it
    provider: str                       # "ollama" or "groq"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None  # "stop", "length", etc.
    raw_response: Optional[dict] = field(default=None, repr=False)

    @property
    def total_tokens_used(self) -> int:
        """Safe token count (returns 0 if not reported by provider)."""
        return self.total_tokens or 0


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider must implement:
        - generate(): text generation
        - health_check(): verify the provider is reachable
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt:        The user prompt / query.
            system_prompt: Optional system instruction.
            temperature:   Sampling temperature (lower = more deterministic).
                           0.1 is recommended for factual RAG tasks.
            max_tokens:    Maximum tokens to generate.
            **kwargs:      Provider-specific parameters.

        Returns:
            LLMResponse with the generated content and metadata.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the provider is reachable and functioning.

        Returns:
            True if healthy, False otherwise.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""
        ...

    @property
    @abstractmethod
    def active_model(self) -> str:
        """Currently active model name."""
        ...
