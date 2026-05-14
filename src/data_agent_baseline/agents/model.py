from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

try:  # OpenAI SDK versions differ slightly on exported error classes.
    from openai import InternalServerError
except ImportError:  # pragma: no cover - compatibility fallback
    InternalServerError = None  # type: ignore[assignment]


TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_ERROR_PATTERNS = (
    "429",
    "bad gateway",
    "connection",
    "expecting value",
    "gateway timeout",
    "internal server error",
    "limit_requests",
    "rate limit",
    "server disconnected",
    "service unavailable",
    "temporarily",
    "timeout",
    "too many requests",
)


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value.strip())


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value.strip())


def _is_transient_model_error(exc: BaseException) -> bool:
    transient_types: tuple[type[BaseException], ...]
    if InternalServerError is None:
        transient_types = (APIConnectionError, APITimeoutError, RateLimitError)
    else:
        transient_types = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

    if isinstance(exc, transient_types):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in TRANSIENT_HTTP_STATUS_CODES:
        return True

    message = str(exc).lower()
    return any(pattern in message for pattern in TRANSIENT_ERROR_PATTERNS)


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        max_retries: int | None = None,
        retry_base_delay_seconds: float | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max(
            max_retries if max_retries is not None else _env_int("MODEL_MAX_RETRIES", 3),
            0,
        )
        self.retry_base_delay_seconds = max(
            retry_base_delay_seconds
            if retry_base_delay_seconds is not None
            else _env_float("MODEL_RETRY_BASE_DELAY_SECONDS", 1.0),
            0.0,
        )

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        request_messages = [{"role": message.role, "content": message.content} for message in messages]
        response = None
        for attempt_index in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    temperature=self.temperature,
                )
                break
            except Exception as exc:
                if attempt_index >= self.max_retries or not _is_transient_model_error(exc):
                    raise RuntimeError(f"Model request failed: {exc}") from exc
                delay_seconds = min(self.retry_base_delay_seconds * (2**attempt_index), 8.0)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

        if response is None:
            raise RuntimeError("Model request failed without returning a response.")

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
