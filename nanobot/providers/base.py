"""Base LLM provider interface."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
import json
from typing import Any

from loguru import logger


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.
    thinking_blocks: list[dict[str, Any]] | None = None  # Anthropic extended thinking
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    def to_debug_dict(self) -> dict[str, Any]:
        """Serialize response for debug logging."""
        return {
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in self.tool_calls
            ],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "reasoning_content": self.reasoning_content,
            "thinking_blocks": self.thinking_blocks,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation parameters for LLM calls."""

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


@dataclass
class AppServerTurnResult:
    """Structured result returned from a Codex App Server turn."""

    thread_id: str
    turn_id: str
    final_text: str
    tools_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """
    
    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalize empty assistant content for provider compatibility.

        Some backends reject empty-string assistant content, while others reject
        assistant tool-call messages when content is omitted. We normalize both
        cases to `content=None` for assistant turns.
        """
        sanitized: list[dict[str, Any]] = []
        for message in messages:
            msg = dict(message)
            if msg.get("role") == "assistant":
                if msg.get("content") == "":
                    msg["content"] = None
                elif "content" not in msg and msg.get("tool_calls"):
                    msg["content"] = None
            sanitized.append(msg)
        return sanitized

    @staticmethod
    def _log_response_debug(response: LLMResponse, model: str | None = None) -> None:
        """Emit full LLM response payload for debugging."""
        model_tag = f" [{model}]" if model else ""
        try:
            payload = json.dumps(response.to_debug_dict(), ensure_ascii=False)
        except Exception:
            payload = str(response)
        logger.debug(f"LLM response{model_tag}: {payload}")

    @staticmethod
    def _is_transient_error(response: LLMResponse) -> bool:
        if response.finish_reason != "error":
            return False
        content = (response.content or "").lower()
        transient_markers = (
            "429",
            "rate limit",
            "rate-limit",
            "timeout",
            "temporarily unavailable",
            "temporary",
            "try again",
        )
        return any(marker in content for marker in transient_markers)

    @staticmethod
    def _strip_images_for_retry(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        stripped_any = False
        sanitized: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                sanitized.append(dict(message))
                continue

            new_blocks: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "image_url":
                    new_blocks.append(dict(block))
                    continue
                meta = block.get("_meta")
                placeholder = "[image omitted]"
                if isinstance(meta, dict):
                    path = meta.get("path")
                    if isinstance(path, str) and path:
                        placeholder = f"[image: {path}]"
                new_blocks.append({"type": "text", "text": placeholder})
                stripped_any = True

            sanitized.append({**message, "content": new_blocks})

        return sanitized, stripped_any

    def _chat_kwargs_for_signature(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return only kwargs accepted by the concrete ``chat()`` implementation."""
        try:
            signature = inspect.signature(self.chat)
        except (TypeError, ValueError):
            return kwargs

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs

        allowed = {
            name
            for name, param in signature.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {key: value for key, value in kwargs.items() if key in allowed}

    async def chat_with_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Run ``chat()`` with generation defaults and a small retry policy."""
        resolved_max_tokens = max_tokens if max_tokens is not None else self.generation.max_tokens
        resolved_temperature = (
            temperature if temperature is not None else self.generation.temperature
        )
        resolved_reasoning = (
            reasoning_effort
            if reasoning_effort is not None
            else self.generation.reasoning_effort
        )

        call_kwargs = {
            "messages": messages,
            "tools": tools,
            "model": model,
            "max_tokens": resolved_max_tokens,
            "temperature": resolved_temperature,
            "reasoning_effort": resolved_reasoning,
        }

        delays = (1, 2, 4)
        last_response: LLMResponse | None = None

        for idx in range(len(delays) + 1):
            try:
                response = await self.chat(**self._chat_kwargs_for_signature(call_kwargs))
            except asyncio.CancelledError:
                raise

            last_response = response
            if not self._is_transient_error(response):
                break
            if idx >= len(delays):
                return response
            await asyncio.sleep(delays[idx])

        if last_response is None:
            return LLMResponse(content="No response", finish_reason="error")

        if last_response.finish_reason != "error":
            return last_response

        retry_messages, stripped_any = self._strip_images_for_retry(messages)
        if not stripped_any:
            return last_response

        retry_kwargs = {**call_kwargs, "messages": retry_messages}
        try:
            return await self.chat(**self._chat_kwargs_for_signature(retry_kwargs))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return LLMResponse(content=str(exc), finish_reason="error")

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            reasoning_effort: Optional reasoning intensity hint (provider-specific).
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @property
    def uses_app_server(self) -> bool:
        """Whether the provider routes primary turns through Codex App Server."""
        return False

    @property
    def supports_native_web_search(self) -> bool:
        """Whether the provider already exposes native web search."""
        return False

    async def run_app_server_turn(self, **_: Any) -> AppServerTurnResult:
        """Run a turn through the provider's App Server runtime."""
        raise NotImplementedError("This provider does not support App Server turns")

    async def aclose(self) -> None:
        """Release provider resources."""
        return None
