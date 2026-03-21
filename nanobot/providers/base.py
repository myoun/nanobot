"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
