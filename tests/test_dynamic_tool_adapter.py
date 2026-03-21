from __future__ import annotations

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


class EchoTool(Tool):
    def __init__(self, result):
        self._result = result

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo text back"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> object:
        return self._result


def test_tool_to_dynamic_tool_spec_matches_app_server_shape() -> None:
    tool = EchoTool("hello")

    assert tool.to_dynamic_tool_spec() == {
        "name": "echo",
        "description": "Echo text back",
        "inputSchema": tool.parameters,
    }
    assert tool.to_dynamic_tool_spec(defer_loading=True) == {
        "name": "echo",
        "description": "Echo text back",
        "inputSchema": tool.parameters,
        "deferLoading": True,
    }


def test_tool_to_dynamic_tool_call_response_normalizes_results() -> None:
    tool = EchoTool("hello")

    assert tool.to_dynamic_tool_call_response("hello") == {
        "contentItems": [{"type": "inputText", "text": "hello"}],
        "success": True,
    }

    structured = tool.to_dynamic_tool_call_response(
        [
            {"type": "inputText", "text": "first"},
            {"type": "inputImage", "imageUrl": "https://example.invalid/image.png"},
        ]
    )
    assert structured == {
        "contentItems": [
            {"type": "inputText", "text": "first"},
            {"type": "inputImage", "imageUrl": "https://example.invalid/image.png"},
        ],
        "success": True,
    }

    json_like = tool.to_dynamic_tool_call_response({"answer": 42})
    assert json_like == {
        "contentItems": [{"type": "inputText", "text": '{"answer": 42}'}],
        "success": True,
    }


@pytest.mark.asyncio
async def test_registry_dynamic_helpers_and_execute_dynamic() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool({"answer": 42}))

    assert registry.get_dynamic_tool_specs() == [
        {
            "name": "echo",
            "description": "Echo text back",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        }
    ]

    response = await registry.execute_dynamic("echo", {"text": "hello"})
    assert response == {
        "contentItems": [{"type": "inputText", "text": '{"answer": 42}'}],
        "success": True,
    }

    missing = await registry.execute_dynamic("missing", {})
    assert missing["success"] is False
    assert missing["contentItems"] == [
        {
            "type": "inputText",
            "text": "Error: Tool 'missing' not found. Available: echo",
        }
    ]
