"""
src/generation/ollama_provider.py

Ollama LLM Provider — primary provider for local development.

Ollama runs Llama models locally on Apple Silicon (MPS).
No API key required. Models are managed by the Ollama daemon.

Models used in this project:
    - llama3.2:3b  (~2 GB) — development and testing
    - llama3.1:8b  (~5 GB) — final evaluation runs

Before using this provider:
    1. Install Ollama: https://ollama.ai/download
    2. Pull models:
       ollama pull llama3.2:3b
       ollama pull llama3.1:8b
    3. Verify Ollama is running:
       ollama list
"""
from __future__ import annotations

import time
from typing import Optional

import ollama
from ollama import ResponseError

from config import get_logger, settings
from src.generation.llm_provider import LLMProvider, LLMResponse

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """
    LLM provider using the local Ollama server.

    Ollama manages model loading and Apple MPS acceleration automatically.
    The Python `ollama` library communicates with the Ollama daemon via HTTP.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model:    Model name (defaults to settings.ollama_dev_model).
            base_url: Ollama server URL (defaults to settings.ollama_base_url).
        """
        self._model = model or settings.ollama_dev_model
        self._base_url = base_url or settings.ollama_base_url
        self._client = ollama.Client(host=self._base_url)
        logger.info(
            "OllamaProvider initialized — model={model}, url={url}",
            model=self._model,
            url=self._base_url,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def active_model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using the local Ollama model.

        The prompt is formatted as a chat message for instruction-tuned models.
        Temperature is intentionally low (0.1) for factual, deterministic output.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "Sending generation request to Ollama — model={model}, tokens={tokens}",
            model=self._model,
            tokens=max_tokens,
        )

        start_time = time.monotonic()

        try:
            raw = self._client.chat(
                model=self._model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs,
                },
            )
        except ResponseError as e:
            logger.error(
                "Ollama ResponseError — model={model}, error={error}",
                model=self._model,
                error=str(e),
            )
            raise RuntimeError(
                f"Ollama generation failed for model '{self._model}'. "
                f"Is the model pulled? Run: ollama pull {self._model}\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            logger.error(
                "Unexpected Ollama error — {error}",
                error=str(e),
            )
            raise

        elapsed = time.monotonic() - start_time
        content = raw.message.content or ""

        logger.debug(
            "Generation complete — elapsed={elapsed:.2f}s, chars={chars}",
            elapsed=elapsed,
            chars=len(content),
        )

        # Extract token counts if Ollama reports them
        prompt_tokens = getattr(raw, "prompt_eval_count", None)
        completion_tokens = getattr(raw, "eval_count", None)
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=getattr(raw, "done_reason", None),
            raw_response=raw.model_dump() if hasattr(raw, "model_dump") else None,
        )

    def health_check(self) -> bool:
        """
        Verify that the Ollama server is running and the model is available.

        Returns:
            True if Ollama is reachable and the model is listed.
        """
        try:
            models_response = self._client.list()
            available_models = [m.model for m in models_response.models]
            # Strip tag for comparison (e.g., "llama3.2:3b" matches "llama3.2:3b")
            is_available = any(
                self._model in m or m in self._model
                for m in available_models
            )
            if not is_available:
                logger.warning(
                    "Model '{model}' not found in Ollama. "
                    "Run: ollama pull {model}. "
                    "Available: {available}",
                    model=self._model,
                    available=available_models,
                )
            else:
                logger.info(
                    "Ollama health check passed — model={model}",
                    model=self._model,
                )
            return is_available
        except Exception as e:
            logger.error(
                "Ollama health check failed — is Ollama running? Error: {error}",
                error=str(e),
            )
            return False
