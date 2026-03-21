"""Tool registry for dynamic tool management."""

import json
from typing import Any

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    def get_dynamic_tool_specs(self, *, defer_loading: bool = False) -> list[dict[str, Any]]:
        """Get all tool definitions in Codex App Server dynamic tool format."""
        return [tool.to_dynamic_tool_spec(defer_loading=defer_loading) for tool in self._tools.values()]

    @staticmethod
    def _normalize_result_content_items(result: Any) -> list[dict[str, Any]]:
        """Normalize a tool result into App Server content items."""
        if result is None:
            return []

        if isinstance(result, dict):
            item_type = result.get("type")
            if item_type in {"inputText", "inputImage"}:
                return [result]

        if isinstance(result, (list, tuple)):
            items: list[dict[str, Any]] = []
            for item in result:
                items.extend(Tool._to_dynamic_content_items(item))
            return items

        if isinstance(result, bytes):
            text = result.decode("utf-8", errors="replace")
        elif isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, ensure_ascii=False, sort_keys=True)

        return [{"type": "inputText", "text": text}]

    def to_dynamic_tool_call_response(self, result: Any, *, success: bool = True) -> dict[str, Any]:
        """Wrap a tool result in an App Server dynamic tool response payload."""
        return {
            "contentItems": self._normalize_result_content_items(result),
            "success": success,
        }

    async def execute_dynamic(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and package the result as an App Server dynamic tool response."""
        tool = self._tools.get(name)
        if not tool:
            return self.to_dynamic_tool_call_response(
                f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}",
                success=False,
            )

        try:
            errors = tool.validate_params(params)
            if errors:
                return self.to_dynamic_tool_call_response(
                    f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors),
                    success=False,
                )
            result = await tool.execute(**params)
            success = not (isinstance(result, str) and result.startswith("Error"))
            return self.to_dynamic_tool_call_response(result, success=success)
        except Exception as e:
            return self.to_dynamic_tool_call_response(
                f"Error executing {name}: {str(e)}",
                success=False,
            )

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
