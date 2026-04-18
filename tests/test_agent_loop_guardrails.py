import asyncio
import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.complete import CompleteTaskTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.report import ReportToUserTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.openai_codex_provider import _consume_sse


class SequenceProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__(api_key=None, api_base=None)
        self._responses = responses
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]

    def get_default_model(self) -> str:
        return "test-model"


class DummyExecTool(Tool):
    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "dummy exec"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(self, command: str, **kwargs: Any) -> str:
        return "ok"


class DummyFailedExecTool(Tool):
    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "dummy exec that fails"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(self, command: str, **kwargs: Any) -> str:
        return "error: command failed\nexit code: 1"


class FakeSSEStream:
    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _data_line(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}"


def _make_loop(workspace: Path, provider: LLMProvider, max_iterations: int = 4) -> AgentLoop:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        max_iterations=max_iterations,
    )
    return loop


@pytest.mark.asyncio
async def test_loop_requires_complete_task_for_turn_end(tmp_path: Path) -> None:
    provider = SequenceProvider(
        responses=[
            LLMResponse(content="I think this is done without any tool call."),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1|fc_1",
                        name="complete_task",
                        arguments={
                            "final_answer": "done",
                            "artifacts": [],
                            "evidence": [],
                            "actions_taken": [],
                        },
                    )
                ],
            ),
        ]
    )
    loop = _make_loop(tmp_path, provider)
    loop.tools = ToolRegistry()
    loop.tools.register(CompleteTaskTool())

    result, tools_used, _ = await loop._run_agent_loop(
        initial_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
    )

    assert result == "done"
    assert provider.calls == 2
    assert tools_used == ["complete_task"]


@pytest.mark.asyncio
async def test_complete_task_without_tool_attempt_is_allowed(tmp_path: Path) -> None:
    provider = SequenceProvider(
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1|fc_1",
                        name="complete_task",
                        arguments={
                            "final_answer": "답변 완료",
                            "artifacts": [],
                            "evidence": [],
                            "actions_taken": [],
                        },
                    )
                ],
            ),
            LLMResponse(content="should not be reached"),
        ]
    )
    loop = _make_loop(tmp_path, provider, max_iterations=6)
    loop.tools = ToolRegistry()
    loop.tools.register(DummyExecTool())
    loop.tools.register(CompleteTaskTool())

    result, tools_used, _ = await loop._run_agent_loop(
        initial_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "안녕?"},
        ]
    )

    assert result == "답변 완료"
    assert provider.calls == 1
    assert tools_used == ["complete_task"]


@pytest.mark.asyncio
async def test_complete_task_allowed_after_failed_tool_attempt(tmp_path: Path) -> None:
    provider = SequenceProvider(
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1|fc_1",
                        name="exec",
                        arguments={"command": "apt-get remove -y yt-dlp"},
                    )
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_2|fc_2",
                        name="complete_task",
                        arguments={
                            "final_answer": "삭제 시도했지만 권한 정책으로 실패했습니다.",
                            "artifacts": [],
                            "evidence": [],
                            "actions_taken": [],
                        },
                    )
                ],
            ),
        ]
    )
    loop = _make_loop(tmp_path, provider, max_iterations=6)
    loop.tools = ToolRegistry()
    loop.tools.register(DummyFailedExecTool())
    loop.tools.register(CompleteTaskTool())

    result, tools_used, _ = await loop._run_agent_loop(
        initial_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "yt-dlp 지워줘"},
        ]
    )

    assert result == "삭제 시도했지만 권한 정책으로 실패했습니다."
    assert provider.calls == 2
    assert tools_used == ["exec", "complete_task"]


@pytest.mark.asyncio
async def test_message_tool_blocks_text_only_to_current_chat() -> None:
    sent: list[OutboundMessage] = []

    async def _send_callback(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(
        send_callback=_send_callback,
        default_channel="telegram",
        default_chat_id="1234",
    )

    result = await tool.execute(content="progress update")
    assert "Text-only message to the current chat is blocked" in result
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_allows_media_to_current_chat() -> None:
    sent: list[OutboundMessage] = []

    async def _send_callback(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(
        send_callback=_send_callback,
        default_channel="telegram",
        default_chat_id="1234",
    )

    result = await tool.execute(content="done", media=["/tmp/sample.png"])
    assert "Message sent to telegram:1234 with 1 media item(s)" in result
    assert len(sent) == 1
    assert sent[0].media == ["/tmp/sample.png"]


@pytest.mark.asyncio
async def test_report_to_user_tool_allows_text_to_current_chat() -> None:
    sent: list[OutboundMessage] = []

    async def _send_callback(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = ReportToUserTool(
        send_callback=_send_callback,
        default_channel="telegram",
        default_chat_id="1234",
    )

    result = await tool.execute(content="진행 상황 공유")
    assert "Progress update sent to telegram:1234" in result
    assert len(sent) == 1
    assert sent[0].content == "진행 상황 공유"
    assert sent[0].media == []


@pytest.mark.asyncio
async def test_terminal_no_tool_text_retries_are_bounded_without_heuristics(tmp_path: Path) -> None:
    provider = SequenceProvider(
        responses=[
            LLMResponse(
                content=(
                    "해당 요청은 정책상 진행할 수 없습니다. 대신 합법 사이트 URL을 보내주세요."
                )
            ),
            LLMResponse(
                content=(
                    "해당 요청은 정책상 진행할 수 없습니다. 대신 합법 사이트 URL을 보내주세요."
                )
            ),
            LLMResponse(
                content=(
                    "해당 요청은 정책상 진행할 수 없습니다. 대신 합법 사이트 URL을 보내주세요."
                )
            ),
            LLMResponse(content="should not be reached"),
        ]
    )
    loop = _make_loop(tmp_path, provider, max_iterations=8)
    loop.tools = ToolRegistry()
    loop.tools.register(CompleteTaskTool())

    result, tools_used, _ = await loop._run_agent_loop(
        initial_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "이 사이트 스크린샷 찍어 보내줘"},
        ]
    )

    assert result is not None
    assert "정책상" in result
    assert provider.calls == 3
    assert tools_used == []


@pytest.mark.asyncio
async def test_no_tool_text_retries_are_bounded(tmp_path: Path) -> None:
    provider = SequenceProvider(
        responses=[
            LLMResponse(content="Understood. I will execute and verify."),
            LLMResponse(content="Understood. I will execute and verify."),
            LLMResponse(content="Understood. I will execute and verify."),
            LLMResponse(content="should not be reached"),
        ]
    )
    loop = _make_loop(tmp_path, provider, max_iterations=10)
    loop.tools = ToolRegistry()
    loop.tools.register(CompleteTaskTool())

    result, tools_used, _ = await loop._run_agent_loop(
        initial_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "velog.io 메인 화면 스크린샷 찍어"},
        ]
    )

    assert result == "Understood. I will execute and verify."
    assert provider.calls == 3
    assert tools_used == []


@pytest.mark.asyncio
async def test_empty_no_tool_retries_are_bounded(tmp_path: Path) -> None:
    provider = SequenceProvider(
        responses=[
            LLMResponse(content=""),
            LLMResponse(content=""),
            LLMResponse(content=""),
            LLMResponse(content=""),
            LLMResponse(content="should not be reached"),
        ]
    )
    loop = _make_loop(tmp_path, provider, max_iterations=10)
    loop.tools = ToolRegistry()
    loop.tools.register(CompleteTaskTool())

    result, tools_used, _ = await loop._run_agent_loop(
        initial_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "스크린샷 찍어"},
        ]
    )

    assert result == AgentLoop._NO_TOOL_FALLBACK
    assert provider.calls == 4
    assert tools_used == []


@pytest.mark.asyncio
async def test_codex_sse_message_done_text_is_parsed() -> None:
    lines = [
        _data_line(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "hello"},
                        {"type": "output_text", "text": " world"},
                    ],
                },
            }
        ),
        "",
        _data_line({"type": "response.completed", "response": {"status": "completed"}}),
        "",
    ]
    response = FakeSSEStream(lines)

    content, tool_calls, finish_reason, metadata = await _consume_sse(response)  # type: ignore[arg-type]

    assert content == "hello world"
    assert tool_calls == []
    assert finish_reason == "stop"
    assert metadata == {}


@pytest.mark.asyncio
async def test_codex_sse_output_text_done_without_delta_is_parsed() -> None:
    lines = [
        _data_line({"type": "response.output_text.done", "text": "final chunk"}),
        "",
        _data_line({"type": "response.completed", "response": {"status": "completed"}}),
        "",
    ]
    response = FakeSSEStream(lines)

    content, tool_calls, finish_reason, metadata = await _consume_sse(response)  # type: ignore[arg-type]

    assert content == "final chunk"
    assert tool_calls == []
    assert finish_reason == "stop"
    assert metadata == {}


@pytest.mark.asyncio
async def test_codex_sse_flushes_last_event_without_blank_line() -> None:
    lines = [
        _data_line({"type": "response.output_text.delta", "delta": "tail"}),
    ]
    response = FakeSSEStream(lines)

    content, tool_calls, finish_reason, metadata = await _consume_sse(response)  # type: ignore[arg-type]

    assert content == "tail"
    assert tool_calls == []
    assert finish_reason == "stop"
    assert metadata == {}


@pytest.mark.asyncio
async def test_process_direct_message_is_serialized(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, SequenceProvider([LLMResponse(content="ok")]))

    state = {"inflight": 0, "max_inflight": 0}

    async def fake_process(self, msg, session_key=None):
        state["inflight"] += 1
        state["max_inflight"] = max(state["max_inflight"], state["inflight"])
        await asyncio.sleep(0.05)
        state["inflight"] -= 1
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=msg.content)

    loop._process_message = types.MethodType(fake_process, loop)

    first, second = await asyncio.gather(
        loop.process_direct_message("first", session_key="s1", channel="cli", chat_id="a"),
        loop.process_direct_message("second", session_key="s2", channel="cli", chat_id="b"),
    )

    assert state["max_inflight"] == 1
    assert first is not None
    assert second is not None
    assert first.content == "first"
    assert second.content == "second"


@pytest.mark.asyncio
async def test_process_direct_message_forwards_metadata(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, SequenceProvider([LLMResponse(content="ok")]))
    captured: dict[str, Any] = {}

    async def fake_process(self, msg, session_key=None):
        captured["metadata"] = msg.metadata
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=msg.content)

    loop._process_message = types.MethodType(fake_process, loop)

    result = await loop.process_direct_message(
        "topic test",
        session_key="s1",
        channel="telegram",
        chat_id="-1001",
        metadata={"message_thread_id": 3},
    )

    assert result is not None
    assert captured["metadata"] == {"message_thread_id": 3}


@pytest.mark.asyncio
async def test_system_callback_uses_metadata_session_id(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, SequenceProvider([LLMResponse(content="unused")]))

    async def fake_run_agent_loop(self, initial_messages, **kwargs):  # type: ignore[no-untyped-def]
        return "callback persisted", [], {}

    loop._run_agent_loop = types.MethodType(fake_run_agent_loop, loop)

    conversation_key = "web:callback-chat"
    _, initial = loop.sessions.get_or_create_for_conversation(conversation_key)
    origin_session_id = str(initial["id"])
    origin_session_key = str(initial["key"])
    created = loop.sessions.create_session(conversation_key, title="Active", switch_to=True)
    active_session_key = str(created["key"])

    response = await loop._process_message(
        InboundMessage(
            channel="system",
            sender_id="system",
            chat_id=conversation_key,
            content="system callback",
            metadata={"session_id": origin_session_id},
        )
    )

    assert response is not None
    assert response.channel == "web"
    assert response.chat_id == "callback-chat"
    assert response.content == "callback persisted"

    origin_session = loop.sessions.get_or_create(origin_session_key)
    active_session = loop.sessions.get_or_create(active_session_key)
    assert any(
        m.get("role") == "user" and "[System: system]" in str(m.get("content") or "")
        for m in origin_session.messages
    )
    assert any(
        m.get("role") == "assistant" and m.get("content") == "callback persisted"
        for m in origin_session.messages
    )
    assert not any(
        m.get("role") == "user" and "[System: system]" in str(m.get("content") or "")
        for m in active_session.messages
    )


@pytest.mark.asyncio
async def test_fixed_new_clears_current_session_only(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, SequenceProvider([LLMResponse(content="unused")]))

    fixed_key = "cli:direct"
    fixed_session = loop.sessions.get_or_create(fixed_key)
    fixed_session.add_message("user", "hello")
    fixed_session.add_message("assistant", "world")
    loop.sessions.save(fixed_session)

    conversation_key = "cli:direct"
    loop.sessions.get_or_create_for_conversation(conversation_key)
    before_snapshot = loop.sessions.list_conversation_sessions(conversation_key)
    loop._run_serialized_consolidation = AsyncMock(return_value=True)  # type: ignore[method-assign]

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="/new",
        ),
        session_key=fixed_key,
    )

    assert response is not None
    assert "Cleared fixed session history" in response.content

    fixed_after = loop.sessions.get_or_create(fixed_key)
    assert fixed_after.messages == []

    after_snapshot = loop.sessions.list_conversation_sessions(conversation_key)
    assert after_snapshot["active_session_id"] == before_snapshot["active_session_id"]
    assert len(after_snapshot["sessions"]) == len(before_snapshot["sessions"])


@pytest.mark.asyncio
async def test_fixed_session_command_does_not_switch_conversation_session(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, SequenceProvider([LLMResponse(content="unused")]))

    fixed_key = "cli:direct"
    conversation_key = "cli:direct"
    loop.sessions.get_or_create_for_conversation(conversation_key)
    target = loop.sessions.create_session(conversation_key, title="Other", switch_to=False)
    before_snapshot = loop.sessions.list_conversation_sessions(conversation_key)

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=f"/session switch {target['id']}",
        ),
        session_key=fixed_key,
    )

    assert response is not None
    assert "unavailable in fixed-session mode" in response.content

    after_snapshot = loop.sessions.list_conversation_sessions(conversation_key)
    assert after_snapshot["active_session_id"] == before_snapshot["active_session_id"]


@pytest.mark.asyncio
async def test_fixed_help_shows_fixed_session_commands(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, SequenceProvider([LLMResponse(content="unused")]))

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="/help",
        ),
        session_key="cli:direct",
    )

    assert response is not None
    assert "fixed-session mode" in response.content
    assert "/new - Clear current fixed session history" in response.content
    assert "/toolhint - Toggle tool usage hints for this session" in response.content


@pytest.mark.asyncio
async def test_toolhint_command_toggles_session_override(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, SequenceProvider([LLMResponse(content="unused")]))

    disable_response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="toolhint-toggle",
            content="/toolhint off",
        )
    )

    assert disable_response is not None
    assert disable_response.content == "Disabled tool hints for this session."

    session = loop.sessions.get_active_session("cli:toolhint-toggle")
    assert session.metadata["tool_hints_enabled"] is False

    status_response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="toolhint-toggle",
            content="/toolhint",
        )
    )

    assert status_response is not None
    assert "Tool hints: disabled" in status_response.content
    assert "Session override: disabled" in status_response.content
    assert "Usage: /toolhint on | /toolhint off | /toolhint reset" in status_response.content

    reset_response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="toolhint-toggle",
            content="/toolhint reset",
        )
    )

    assert reset_response is not None
    assert reset_response.content == "Cleared the session tool-hint override. Using default: enabled."

    session = loop.sessions.get_active_session("cli:toolhint-toggle")
    assert "tool_hints_enabled" not in session.metadata


@pytest.mark.asyncio
async def test_command_router_accepts_telegram_priority_bot_mention() -> None:
    router = CommandRouter()

    async def handler(ctx: CommandContext) -> OutboundMessage:
        return OutboundMessage(channel="telegram", chat_id="chat", content=f"handled {ctx.raw}")

    router.priority("/restart", handler)
    ctx = CommandContext(
        msg=InboundMessage(channel="telegram", sender_id="user", chat_id="chat", content="/restart@helios001bot"),
        session=None,
        key="telegram:chat",
        raw="/restart@helios001bot",
        loop=None,
    )

    assert router.is_priority("/restart@helios001bot") is True
    result = await router.dispatch_priority(ctx)

    assert result is not None
    assert result.content == "handled /restart"


@pytest.mark.asyncio
async def test_command_router_accepts_telegram_prefix_bot_mention() -> None:
    router = CommandRouter()

    async def handler(ctx: CommandContext) -> OutboundMessage:
        return OutboundMessage(channel="telegram", chat_id="chat", content=ctx.args)

    router.prefix("/model ", handler)
    ctx = CommandContext(
        msg=InboundMessage(channel="telegram", sender_id="user", chat_id="chat", content="/model@helios001bot gpt-5"),
        session=None,
        key="telegram:chat",
        raw="/model@helios001bot gpt-5",
        loop=None,
    )

    result = await router.dispatch(ctx)

    assert result is not None
    assert result.content == "gpt-5"
