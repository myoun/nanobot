from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.openai_codex_app_server_provider import OpenAICodexAppServerProvider


class FakeAppServerClient:
    def __init__(self) -> None:
        self.ensure_thread_calls: list[dict[str, Any]] = []
        self.run_turn_calls: list[dict[str, Any]] = []
        self.closed = False

    async def ensure_thread(self, **kwargs: Any) -> str:
        self.ensure_thread_calls.append(kwargs)
        return "thread-app-123"

    async def run_turn(self, **kwargs: Any) -> tuple[str, str, list[str], dict[str, Any]]:
        self.run_turn_calls.append(kwargs)
        event_callback = kwargs.get("event_callback")
        if event_callback is not None:
            await event_callback(
                {
                    "type": "agent_delta",
                    "thread_id": "thread-app-123",
                    "turn_id": "turn-app-456",
                    "item_id": "msg-1",
                    "phase": "analysis",
                    "delta": "agent-browser",
                }
            )
            await event_callback(
                {
                    "type": "agent_delta",
                    "thread_id": "thread-app-123",
                    "turn_id": "turn-app-456",
                    "item_id": "msg-1",
                    "phase": "analysis",
                    "delta": " ",
                }
            )
            await event_callback(
                {
                    "type": "agent_delta",
                    "thread_id": "thread-app-123",
                    "turn_id": "turn-app-456",
                    "item_id": "msg-1",
                    "phase": "analysis",
                    "delta": "스킬로",
                }
            )
            await event_callback(
                {
                    "type": "agent_delta",
                    "thread_id": "thread-app-123",
                    "turn_id": "turn-app-456",
                    "item_id": "msg-1",
                    "phase": "analysis",
                    "delta": " ",
                }
            )
            await event_callback(
                {
                    "type": "agent_delta",
                    "thread_id": "thread-app-123",
                    "turn_id": "turn-app-456",
                    "item_id": "msg-1",
                    "phase": "analysis",
                    "delta": "직접 확인",
                }
            )
            await event_callback(
                {
                    "type": "agent_delta",
                    "thread_id": "thread-app-123",
                    "turn_id": "turn-app-456",
                    "item_id": "msg-1",
                    "phase": "analysis",
                    "delta": "한다.",
                }
            )
            await event_callback(
                {
                    "type": "tool_call",
                    "thread_id": "thread-app-123",
                    "turn_id": "turn-app-456",
                    "call_id": "call-1",
                    "tool": "echo_tool",
                    "arguments": {"text": "hello"},
                }
            )
            await event_callback(
                {
                    "type": "tool_result",
                    "thread_id": "thread-app-123",
                    "turn_id": "turn-app-456",
                    "call_id": "call-1",
                    "tool": "echo_tool",
                    "success": True,
                    "result_preview": "echo:hello",
                }
            )
        return (
            "turn-app-456",
            "final answer from app server",
            ["echo_tool"],
            {"token_usage": {"total": {"inputTokens": 3210, "outputTokens": 210, "totalTokens": 3420}}},
        )

    async def get_runtime_status(self) -> dict[str, Any]:
        return {
            "account": {
                "authMode": "chatgpt",
                "planType": "pro",
            },
            "config": {
                "model_context_window": 400_000,
                "model_auto_compact_token_limit": 200_000,
            },
            "rate_limits": {
                "primary": {
                    "usedPercent": 42,
                    "windowDurationMins": 300,
                    "resetsAt": 1_900_000_000,
                },
                "secondary": {
                    "usedPercent": 18,
                    "windowDurationMins": 10_080,
                    "resetsAt": 1_900_604_800,
                },
            },
        }

    async def aclose(self) -> None:
        self.closed = True


def _make_loop(workspace: Path, provider: OpenAICodexAppServerProvider) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        max_iterations=2,
    )


@pytest.mark.asyncio
async def test_agent_loop_app_server_branch_stores_thread_and_final_text(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    async def fake_classify(session, user_text):
        return "TASK", "OPTIONAL", "test classifier"

    loop._classify_request = fake_classify  # type: ignore[method-assign]

    try:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="appserver-thread",
                content="hello from app server",
            )
        )

        assert response is not None
        assert response.content == "final answer from app server"
        assert fake_client.ensure_thread_calls
        assert fake_client.run_turn_calls
        assert fake_client.ensure_thread_calls[0]["thread_id"] is None
        assert fake_client.run_turn_calls[0]["thread_id"] == "thread-app-123"
        assert fake_client.run_turn_calls[0]["input_items"][-1]["text"] == (
            "[Current User Message]\nhello from app server"
        )
        progress = await loop.bus.consume_outbound()
        assert progress.content == "agent-browser 스킬로 직접 확인한다."
        assert progress.metadata["_progress"] is True
        tool_hint = await loop.bus.consume_outbound()
        assert tool_hint.content == 'Using tool: echo_tool\n```json\n{\n  "text": "hello"\n}\n```'
        assert tool_hint.metadata["_tool_hint"] is True
        tool_result_hint = await loop.bus.consume_outbound()
        assert tool_result_hint.content == "Tool result: echo_tool\n```text\necho:hello\n```"
        assert tool_result_hint.metadata["_tool_hint"] is True
        assert loop.bus.outbound_size == 0

        session = loop.sessions.get_active_session("cli:appserver-thread")
        assert session.metadata["app_server_thread_id"] == "thread-app-123"
        assert session.metadata["app_server_last_turn_id"] == "turn-app-456"
        assert session.metadata["app_server_token_usage"]["total"]["totalTokens"] == 3420
        assert session.messages[-1]["content"] == "final answer from app server"
        assert session.messages[-1]["tools_used"] == ["echo_tool"]
    finally:
        await loop.close_mcp()


def test_tool_hint_formats_common_tools() -> None:
    assert AgentLoop._app_server_tool_hint(
        "read_file",
        {"path": "/home/myoun/.nanobot/workspace/AGENTS.md"},
    ) == "Using tool: read_file\n```text\n/home/myoun/.nanobot/workspace/AGENTS.md\n```"
    assert AgentLoop._app_server_tool_hint(
        "exec",
        {"command": 'rg -n "heartbeat|HEARTBEAT|cron" /home/myoun/code/nanobot'},
    ) == 'Using tool: exec\n```bash\nrg -n "heartbeat|HEARTBEAT|cron" /home/myoun/code/nanobot\n```'
    assert AgentLoop._app_server_tool_hint(
        "cron",
        {"action": "list"},
    ) == 'Using tool: cron\n```json\n{\n  "action": "list"\n}\n```'


def test_app_server_tool_result_hint_formats_preview() -> None:
    assert AgentLoop._app_server_tool_result_hint(
        "read_file",
        "1| line one\n2| line two",
    ) == "Tool result: read_file\n```text\n1| line one\n2| line two\n```"


def test_tool_detail_block_uses_longer_fence_when_needed() -> None:
    assert AgentLoop._tool_detail_block("print('```')", language="python") == "````python\nprint('```')\n````"


@pytest.mark.asyncio
async def test_agent_loop_app_server_branch_reuses_existing_thread(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    async def fake_classify(session, user_text):
        return "TASK", "OPTIONAL", "test classifier"

    loop._classify_request = fake_classify  # type: ignore[method-assign]

    try:
        first = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="resume-thread",
                content="first turn",
            )
        )
        second = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="resume-thread",
                content="second turn",
            )
        )

        assert first is not None and second is not None
        assert len(fake_client.ensure_thread_calls) == 2
        assert fake_client.ensure_thread_calls[0]["thread_id"] is None
        assert fake_client.ensure_thread_calls[1]["thread_id"] == "thread-app-123"
        assert fake_client.run_turn_calls[0]["thread_id"] == "thread-app-123"
        assert fake_client.run_turn_calls[1]["thread_id"] == "thread-app-123"

        session = loop.sessions.get_active_session("cli:resume-thread")
        assert session.metadata["app_server_thread_id"] == "thread-app-123"
        assert session.messages[-1]["content"] == "final answer from app server"
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_agent_loop_app_server_bootstraps_working_set_for_fresh_thread(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    async def fake_classify(session, user_text):
        return "TASK", "OPTIONAL", "test classifier"

    loop._classify_request = fake_classify  # type: ignore[method-assign]

    session = loop.sessions.get_active_session("cli:bootstrap-working-set")
    session.title = "Bootstrap working set"
    session.summary = "Carry the current migration state into a fresh thread."
    session.add_message("user", "Old turn that should be excluded.")
    session.add_message("assistant", "Old assistant response.", tools_used=["sessions"])
    session.add_message("user", "Turn one to preserve.")
    session.add_message("assistant", "I inspected the migration state.", tools_used=["sessions"])
    session.add_message("user", "Finish the migration rebase plan.")
    session.add_message("assistant", "I will keep the migration constraints in mind.", tools_used=["sessions"])
    session.add_message("user", "Also keep the rollback plan nearby.")
    session.add_message("assistant", "Rollback plan is noted.", tools_used=["sessions"])
    loop.sessions.save(session)

    try:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="bootstrap-working-set",
                content="continue",
            )
        )

        assert response is not None
        input_items = fake_client.run_turn_calls[0]["input_items"]
        working_set_items = [
            item for item in input_items
            if item["type"] == "text" and "[Local Session Working Set]" in item["text"]
        ]
        history_items = [
            item for item in input_items
            if item["type"] == "text" and "[Local Session Bootstrap]" in item["text"]
        ]
        assert working_set_items
        assert history_items
        assert "Also keep the rollback plan nearby." in working_set_items[0]["text"]
        assert "Carry the current migration state into a fresh thread." in working_set_items[0]["text"]
        assert "working_set.md" in working_set_items[0]["text"]
        assert "Old turn that should be excluded." not in history_items[0]["text"]
        assert "Turn one to preserve." in history_items[0]["text"]
        assert "Finish the migration rebase plan." in history_items[0]["text"]
        assert "Also keep the rollback plan nearby." in history_items[0]["text"]
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_rebase_clears_remote_thread_binding_but_keeps_session_history(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    session = loop.sessions.get_active_session("cli:rebase-current")
    session.metadata["app_server_thread_id"] = "thread-app-123"
    session.metadata["app_server_last_turn_id"] = "turn-app-456"
    session.add_message("user", "keep this history")
    loop.sessions.save(session)

    try:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="rebase-current",
                content="/rebase",
            )
        )

        assert response is not None
        assert "fresh Codex thread" in response.content
        refreshed = loop.sessions.get_active_session("cli:rebase-current")
        assert "app_server_thread_id" not in refreshed.metadata
        assert "app_server_last_turn_id" not in refreshed.metadata
        assert refreshed.messages[-1]["content"] == "keep this history"
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_model_command_reports_current_session_model(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    try:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="model-info",
                content="/model",
            )
        )

        assert response is not None
        assert "Current model: openai-codex/gpt-5.1-codex" in response.content
        assert "Session override: none" in response.content
        assert "Usage: /model <name> | /model reset" in response.content
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_status_command_reports_session_limits_and_context_window(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    session = loop.sessions.get_active_session("cli:status-info")
    session.title = "Status test"
    session.metadata["app_server_thread_id"] = "thread-status-1"
    session.metadata["app_server_token_usage"] = {
        "total": {"inputTokens": 12_000, "outputTokens": 800, "totalTokens": 12_800}
    }
    loop.sessions.save(session)

    try:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="status-info",
                content="/status",
            )
        )

        assert response is not None
        assert "Model: openai-codex/gpt-5.1-codex" in response.content
        assert "Routing: enabled" in response.content
        assert "Tool hints: enabled" in response.content
        assert "Session: " in response.content
        assert "Conversation: cli:status-info" in response.content
        assert "App Server thread: thread-status-1" in response.content
        assert "Auth: chatgpt" in response.content
        assert "Plan: pro" in response.content
        assert "5h limit: 42% used" in response.content
        assert "Weekly limit: 18% used" in response.content
        assert "Context left: ~94% (187,200 / 200,000 tokens remaining in auto-compact budget" in response.content
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_toolhint_command_hides_app_server_tool_progress_for_session(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    async def fake_classify(session, user_text):
        return "TASK", "OPTIONAL", "test classifier"

    loop._classify_request = fake_classify  # type: ignore[method-assign]

    try:
        disable_response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="toolhint-progress",
                content="/toolhint off",
            )
        )

        assert disable_response is not None
        assert disable_response.content == "Disabled tool hints for this session."

        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="toolhint-progress",
                content="continue",
            )
        )

        assert response is not None
        assert response.content == "final answer from app server"
        progress = await loop.bus.consume_outbound()
        assert progress.content == "agent-browser 스킬로 직접 확인한다."
        assert progress.metadata["_progress"] is True
        assert loop.bus.outbound_size == 0
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_routing_command_disables_request_classification_for_session(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    async def should_not_run(_session, _user_text):
        raise AssertionError("classification should be bypassed when routing is disabled")

    loop._classify_request = should_not_run  # type: ignore[method-assign]

    try:
        disable_response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="routing-toggle",
                content="/routing off",
            )
        )

        assert disable_response is not None
        assert disable_response.content == "Disabled intent/execution routing for this session."

        routed_response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="routing-toggle",
                content="just continue",
            )
        )

        assert routed_response is not None
        assert routed_response.content == "final answer from app server"
        session = loop.sessions.get_active_session("cli:routing-toggle")
        assert session.metadata["routing_enabled"] is False
        assert session.metadata["last_request_intent"] == "TASK"
        assert session.metadata["last_request_execution"] == "OPTIONAL"
        assert session.metadata["last_request_reason"] == "intent/execution routing disabled for this session"
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_model_command_switches_session_model_without_detaching_thread_binding(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    async def fake_classify(session, user_text):
        return "TASK", "OPTIONAL", "test classifier"

    loop._classify_request = fake_classify  # type: ignore[method-assign]

    session = loop.sessions.get_active_session("cli:model-switch")
    session.metadata["app_server_thread_id"] = "thread-old"
    session.metadata["app_server_last_turn_id"] = "turn-old"
    session.metadata["app_server_model"] = "openai-codex/gpt-5.1-codex"
    loop.sessions.save(session)

    try:
        switch_response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="model-switch",
                content="/model gpt-5.4",
            )
        )

        assert switch_response is not None
        assert switch_response.content == "Switched this session to model openai-codex/gpt-5.4."

        refreshed = loop.sessions.get_active_session("cli:model-switch")
        assert refreshed.metadata["model_override"] == "openai-codex/gpt-5.4"
        assert refreshed.metadata["app_server_thread_id"] == "thread-old"
        assert refreshed.metadata["app_server_last_turn_id"] == "turn-old"
        assert refreshed.metadata["app_server_model"] == "openai-codex/gpt-5.1-codex"

        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="model-switch",
                content="continue with the new model",
            )
        )

        assert response is not None
        assert response.content == "final answer from app server"
        assert fake_client.ensure_thread_calls[-1]["thread_id"] == "thread-old"
        assert fake_client.ensure_thread_calls[-1]["model"] == "gpt-5.4"
        assert fake_client.run_turn_calls[-1]["model"] == "gpt-5.4"

        latest_session = loop.sessions.get_active_session("cli:model-switch")
        assert latest_session.metadata["app_server_model"] == "openai-codex/gpt-5.4"
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_model_command_reset_returns_session_to_default_model(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    session = loop.sessions.get_active_session("cli:model-reset")
    session.metadata["model_override"] = "openai-codex/gpt-5.4"
    session.metadata["app_server_thread_id"] = "thread-old"
    session.metadata["app_server_last_turn_id"] = "turn-old"
    session.metadata["app_server_model"] = "openai-codex/gpt-5.4"
    loop.sessions.save(session)

    try:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="model-reset",
                content="/model reset",
            )
        )

        assert response is not None
        assert response.content == "Reset this session to the default model: openai-codex/gpt-5.1-codex."

        refreshed = loop.sessions.get_active_session("cli:model-reset")
        assert "model_override" not in refreshed.metadata
        assert refreshed.metadata["app_server_thread_id"] == "thread-old"
        assert refreshed.metadata["app_server_last_turn_id"] == "turn-old"
        assert refreshed.metadata["app_server_model"] == "openai-codex/gpt-5.4"
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_app_server_runtime_skips_local_consolidation_tracking(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)
    loop.memory_window = 2

    async def fake_classify(session, user_text):
        return "TASK", "OPTIONAL", "test classifier"

    loop._classify_request = fake_classify  # type: ignore[method-assign]

    def should_not_track(_session):
        raise AssertionError("App Server runtime should not schedule local consolidation")

    loop._track_consolidation_task = should_not_track  # type: ignore[method-assign]

    session = loop.sessions.get_active_session("cli:no-consolidation")
    session.add_message("user", "one")
    session.add_message("assistant", "two")
    session.add_message("user", "three")
    session.add_message("assistant", "four")
    loop.sessions.save(session)

    try:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="no-consolidation",
                content="continue",
            )
        )
        assert response is not None
        assert response.content == "final answer from app server"
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_new_in_app_server_runtime_skips_archive_and_preserves_old_session(
    tmp_path: Path,
) -> None:
    fake_client = FakeAppServerClient()
    provider = OpenAICodexAppServerProvider(
        default_model="openai-codex/gpt-5.1-codex",
        workspace=tmp_path,
        app_server_client=fake_client,  # type: ignore[arg-type]
    )
    loop = _make_loop(tmp_path, provider)

    async def fail_archive(*args, **kwargs):
        raise AssertionError("Conversation /new in App Server runtime should not archive via MEMORY/HISTORY")

    loop._run_serialized_consolidation = fail_archive  # type: ignore[method-assign]

    session = loop.sessions.get_active_session("cli:new-app-server")
    session.add_message("user", "keep old history")
    loop.sessions.save(session)
    old_session_id = session.id

    try:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="new-app-server",
                content="/new",
            )
        )

        assert response is not None
        assert "New session started." in response.content
        snapshot = loop.sessions.list_conversation_sessions("cli:new-app-server")
        assert snapshot["active_session_id"] != old_session_id
        assert len(snapshot["sessions"]) == 2
        old_session = loop.sessions.get_by_id(str(old_session_id))
        assert old_session is not None
        assert old_session.messages[-1]["content"] == "keep old history"
    finally:
        await loop.close_mcp()
