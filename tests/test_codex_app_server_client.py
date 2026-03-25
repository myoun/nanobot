from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.codex_app_server_client import CodexAppServerClient
from nanobot.providers.codex_profile import (
    CodexProfileManager,
    DEFAULT_COMPACT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
)
from nanobot.providers.openai_codex_app_server_provider import OpenAICodexAppServerProvider
from nanobot.utils.helpers import get_data_path


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo input text"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs):
        return f"echo:{kwargs['text']}"


class CompleteTool(Tool):
    @property
    def name(self) -> str:
        return "complete_task"

    @property
    def description(self) -> str:
        return "Legacy completion tool"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "unused"


@pytest.mark.asyncio
async def test_codex_app_server_client_executes_dynamic_tool_and_collects_final_text(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_app_server.py"
    script.write_text(
        """
import json
import sys

thread_id = "thread-1"
turn_id = "turn-1"

for raw in sys.stdin:
    msg = json.loads(raw)
    method = msg.get("method")
    request_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({"id": request_id, "result": {"userAgent": "fake", "platformFamily": "unix", "platformOs": "linux"}}), flush=True)
    elif method == "initialized":
        continue
    elif method == "thread/start":
        print(json.dumps({"id": request_id, "result": {"thread": {"id": thread_id}}}), flush=True)
    elif method == "thread/resume":
        print(json.dumps({"id": request_id, "result": {"thread": {"id": msg["params"]["threadId"]}}}), flush=True)
    elif method == "turn/start":
        print(json.dumps({"id": request_id, "result": {"turn": {"id": turn_id, "status": "inProgress", "items": [], "error": None}}}), flush=True)
        print(json.dumps({"method": "turn/started", "params": {"threadId": thread_id, "turn": {"id": turn_id, "status": "inProgress", "items": [], "error": None}}}), flush=True)
        print(json.dumps({"method": "item/tool/call", "id": 7, "params": {"threadId": thread_id, "turnId": turn_id, "callId": "call-1", "tool": "echo", "arguments": {"text": "hello"}}}), flush=True)
    elif request_id == 7:
        print(json.dumps({"method": "item/completed", "params": {"threadId": thread_id, "turnId": turn_id, "item": {"type": "dynamicToolCall", "id": "call-1", "tool": "echo", "arguments": {"text": "hello"}, "status": "completed", "contentItems": msg["result"]["contentItems"], "success": msg["result"]["success"]}}}), flush=True)
        print(json.dumps({"method": "item/started", "params": {"threadId": thread_id, "turnId": turn_id, "item": {"type": "agentMessage", "id": "msg-1", "text": "", "phase": "final_answer"}}}), flush=True)
        print(json.dumps({"method": "item/agentMessage/delta", "params": {"threadId": thread_id, "turnId": turn_id, "itemId": "msg-1", "delta": "final hello"}}), flush=True)
        print(json.dumps({"method": "item/completed", "params": {"threadId": thread_id, "turnId": turn_id, "item": {"type": "agentMessage", "id": "msg-1", "text": "final hello", "phase": "final_answer"}}}), flush=True)
        print(json.dumps({"method": "thread/tokenUsage/updated", "params": {"threadId": thread_id, "turnId": turn_id, "tokenUsage": {"total": {"totalTokens": 12}}}}), flush=True)
        print(json.dumps({"method": "turn/completed", "params": {"threadId": thread_id, "turn": {"id": turn_id, "status": "completed", "items": [], "error": None}}}), flush=True)
""".strip(),
        encoding="utf-8",
    )

    client = CodexAppServerClient(
        command=[sys.executable, str(script)],
        cwd=tmp_path,
        client_name="pytest",
        client_title="pytest",
        client_version="0",
    )
    thread_id = await client.ensure_thread(
        thread_id=None,
        dynamic_tools=[{
            "name": "echo",
            "description": "Echo input text",
            "inputSchema": EchoTool().parameters,
        }],
        developer_instructions="Use the echo tool before answering.",
        cwd=str(tmp_path),
    )

    async def exec_tool(name: str, args: dict[str, object]) -> dict[str, object]:
        assert name == "echo"
        assert args == {"text": "hello"}
        return {"contentItems": [{"type": "inputText", "text": "echo:hello"}], "success": True}

    events: list[dict[str, object]] = []

    async def on_event(event: dict[str, object]) -> None:
        events.append(event)

    turn_id, final_text, tools_used, metadata = await client.run_turn(
        thread_id=thread_id,
        input_items=[{"type": "text", "text": "Say hello", "text_elements": []}],
        tool_executor=exec_tool,
        event_callback=on_event,
        cwd=str(tmp_path),
    )

    assert thread_id == "thread-1"
    assert turn_id == "turn-1"
    assert final_text == "final hello"
    assert tools_used == ["echo"]
    assert metadata["token_usage"]["total"]["totalTokens"] == 12
    assert [event["type"] for event in events] == ["tool_call", "tool_result", "agent_delta", "token_usage"]
    assert events[0]["tool"] == "echo"
    assert events[2]["delta"] == "final hello"
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_codex_app_server_provider_uses_dynamic_tools_and_filters_complete_task(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(CompleteTool())

    captured: dict[str, object] = {}

    class StubClient:
        async def ensure_thread(self, **kwargs):
            captured["ensure_thread"] = kwargs
            return "thread-xyz"

        async def run_turn(self, **kwargs):
            captured["run_turn"] = kwargs
            tool_executor = kwargs["tool_executor"]
            tool_result = await tool_executor("echo", {"text": "hello"})
            captured["tool_result"] = tool_result
            return "turn-xyz", "provider final", ["echo"], {"token_usage": {"total": {"totalTokens": 9}}}

        async def aclose(self) -> None:
            captured["closed"] = True

    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=StubClient(),  # type: ignore[arg-type]
    )
    result = await provider.run_app_server_turn(
        thread_id=None,
        input_items=[{"type": "text", "text": "hello", "text_elements": []}],
        tools=registry,
        developer_instructions="Use tools.",
        cwd=str(tmp_path),
        exclude_tool_names=["complete_task"],
    )

    ensure_thread = captured["ensure_thread"]
    assert isinstance(ensure_thread, dict)
    dynamic_tool_names = [tool["name"] for tool in ensure_thread["dynamic_tools"]]
    assert dynamic_tool_names == ["echo"]
    assert ensure_thread["sandbox"] == "danger-full-access"
    assert captured["tool_result"] == {
        "contentItems": [{"type": "inputText", "text": "echo:hello"}],
        "success": True,
    }
    assert result.thread_id == "thread-xyz"
    assert result.turn_id == "turn-xyz"
    assert result.final_text == "provider final"
    await provider.aclose()
    assert captured["closed"] is True


@pytest.mark.asyncio
async def test_openai_codex_app_server_provider_can_override_sandbox(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class StubClient:
        async def ensure_thread(self, **kwargs):
            captured["ensure_thread"] = kwargs
            return "thread-xyz"

        async def run_turn(self, **kwargs):
            return "turn-xyz", "provider final", [], {}

        async def aclose(self) -> None:
            return None

    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=StubClient(),  # type: ignore[arg-type]
        sandbox="workspace-write",
    )
    await provider.run_app_server_turn(
        thread_id=None,
        input_items=[{"type": "text", "text": "hello", "text_elements": []}],
        tools=ToolRegistry(),
        developer_instructions="Use tools.",
        cwd=str(tmp_path),
    )

    ensure_thread = captured["ensure_thread"]
    assert isinstance(ensure_thread, dict)
    assert ensure_thread["sandbox"] == "workspace-write"


@pytest.mark.asyncio
async def test_codex_app_server_client_creates_workspace_profile_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOBOT_HOME", str(tmp_path / ".nanobot-home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex-home"))
    script = tmp_path / "fake_init_only.py"
    script.write_text(
        """
import json
import sys

for raw in sys.stdin:
    msg = json.loads(raw)
    if msg.get("method") == "initialize":
        print(json.dumps({"id": msg.get("id"), "result": {"userAgent": "fake", "platformFamily": "unix", "platformOs": "linux"}}), flush=True)
    elif msg.get("method") == "initialized":
        continue
""".strip(),
        encoding="utf-8",
    )

    client = CodexAppServerClient(
        command=[sys.executable, str(script)],
        cwd=tmp_path,
        client_name="pytest",
        client_title="pytest",
        client_version="0",
    )
    await client.ensure_started()

    config_text = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    system_prompt_text = (get_data_path() / "codex" / "system_prompt.md").read_text(encoding="utf-8")
    compact_prompt_text = (get_data_path() / "codex" / "compact_prompt.md").read_text(encoding="utf-8")

    assert "[profiles.nanobot]" in config_text
    assert 'model_instructions_file = "' in config_text
    assert 'experimental_compact_prompt_file = "' in config_text
    assert system_prompt_text == DEFAULT_SYSTEM_PROMPT
    assert compact_prompt_text == DEFAULT_COMPACT_PROMPT
    await client.aclose()


def test_codex_profile_manager_marks_workspace_as_trusted(monkeypatch, tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    manager = CodexProfileManager(workspace)
    manager.ensure_profile()

    global_config = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert f'[projects."{workspace.resolve()}"]' in global_config
    assert 'trust_level = "trusted"' in global_config


def test_codex_profile_manager_upgrades_existing_workspace_trust(monkeypatch, tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        (
            'model = "gpt-5.4"\n\n'
            f'[projects."{workspace.resolve()}"]\n'
            'trust_level = "untrusted"\n'
        ),
        encoding="utf-8",
    )

    manager = CodexProfileManager(workspace)
    manager.ensure_profile()

    global_config = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert f'[projects."{workspace.resolve()}"]' in global_config
    assert 'trust_level = "trusted"' in global_config
    assert 'trust_level = "untrusted"' not in global_config


def test_codex_app_server_default_command_injects_profile() -> None:
    command = CodexAppServerClient._inject_profile(
        ["codex", "app-server", "--listen", "stdio://"],
        "nanobot",
    )
    assert command == ["codex", "--profile", "nanobot", "app-server", "--listen", "stdio://"]


def test_codex_app_server_client_can_skip_workspace_profile_injection(tmp_path: Path) -> None:
    client = CodexAppServerClient(
        command=["codex", "app-server", "--listen", "stdio://"],
        cwd=tmp_path,
        use_workspace_profile=False,
    )

    assert client.command == ["codex", "app-server", "--listen", "stdio://"]
