"""
src/generation/groq_provider.py

Groq API LLM Provider — fallback for final experiments or when offline.

Groq uses custom LPU (Language Processing Unit) hardware for extremely
fast inference. The API is compatible with the OpenAI SDK format.

When to use Groq:
    - Final experimental evaluation (better model quality)
    - When internet is available and Ollama is too slow
    - When testing with llama-3.3-70b (too large for local RAM)

Setup:
    1. Create account at https://console.groq.com (free tier available)
    2. Generate an API key
    3. Set GROQ_API_KEY in .env
    4. Set LLM_PROVIDER=groq in .env
"""
from __future__ import annotations

import time
from typing import Optional

from groq import Groq, APIError, APIConnectionError

from config import get_logger, settings
from src.generation.llm_provider import LLMProvider, LLMResponse

logger = get_logger(__name__)


class GroqProvider(LLMProvider):
    """
    LLM provider using the Groq Cloud API.

    Uses the Groq Python SDK, which follows OpenAI SDK conventions.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Groq provider.

        Args:
            model:   Groq model name (defaults to settings.groq_model).
            api_key: Groq API key (defaults to settings.groq_api_key).

        Raises:
            ValueError: If GROQ_API_KEY is missing.
        """
        self._model = model or settings.groq_model
        resolved_key = api_key or settings.groq_api_key

        if not resolved_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to .env or set LLM_PROVIDER=ollama to use local inference."
            )

        self._client = Groq(api_key=resolved_key)
        logger.info(
            "GroqProvider initialized — model={model}",
            model=self._model,
        )

    @property
    def provider_name(self) -> str:
        return "groq"

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
        Generate a response using the Groq API.

        Uses the chat completions endpoint (OpenAI-compatible format).
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "Sending generation request to Groq — model={model}",
            model=self._model,
        )

        start_time = time.monotonic()

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except APIConnectionError as e:
            logger.error("Groq API connection failed — {error}", error=str(e))
            raise RuntimeError(
                "Cannot connect to Groq API. Check your internet connection. "
                "To use local inference instead: set LLM_PROVIDER=ollama in .env"
            ) from e
        except APIError as e:
            logger.error(
                "Groq API error — status={status}, error={error}",
                status=e.status_code,
                error=str(e),
            )
            raise RuntimeError(f"Groq API error (status {e.status_code}): {e}") from e

        elapsed = time.monotonic() - start_time
        choice = completion.choices[0]
        content = choice.message.content or ""

        logger.debug(
            "Groq generation complete — elapsed={elapsed:.2f}s, tokens={tokens}",
            elapsed=elapsed,
            tokens=completion.usage.total_tokens if completion.usage else 0,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else None,
            completion_tokens=completion.usage.completion_tokens if completion.usage else None,
            total_tokens=completion.usage.total_tokens if completion.usage else None,
            finish_reason=choice.finish_reason,
            raw_response=completion.model_dump() if hasattr(completion, "model_dump") else None,
        )

    def health_check(self) -> bool:
        """
        Verify the Groq API is reachable by listing available models.

        Returns:
            True if the API key is valid and the API is reachable.
        """
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            is_available = self._model in available
            if not is_available:
                logger.warning(
                    "Model '{model}' not in Groq available models: {available}",
                    model=self._model,
                    available=available[:5],
                )
            else:
                logger.info("Groq health check passed — model={model}", model=self._model)
            return is_available
        except Exception as e:
            logger.error("Groq health check failed — {error}", error=str(e))
            return False
