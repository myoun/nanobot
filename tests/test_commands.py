import asyncio
import contextlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
import websockets
from typer.testing import CliRunner

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.discord import DiscordChannel
from nanobot.channels.telegram import TelegramChannel
from nanobot.channels.web import WebChannel
from nanobot.cli.commands import app
from nanobot.config.schema import DiscordConfig, TelegramConfig, WebConfig
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.session.manager import SessionManager

runner = CliRunner()


class _NoopProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        raise AssertionError("chat should not be called for fixed-session command tests")

    def get_default_model(self) -> str:
        return "test-model"


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with (
        patch("nanobot.config.loader.get_config_path") as mock_cp,
        patch("nanobot.config.loader.save_config") as mock_sc,
        patch("nanobot.config.loader.load_config"),
        patch("nanobot.utils.helpers.get_workspace_path") as mock_ws,
    ):
        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


@pytest.mark.asyncio
async def test_web_channel_http_and_ws_roundtrip() -> None:
    bus = MessageBus()
    channel = WebChannel(
        WebConfig(enabled=True, host="127.0.0.1", port=0),
        bus,
    )
    await channel.start()

    running = True

    async def fake_agent() -> None:
        while running:
            inbound = await bus.consume_inbound()
            await bus.publish_outbound(
                OutboundMessage(
                    channel="web",
                    chat_id=inbound.chat_id,
                    content=f"echo:{inbound.content}",
                )
            )

    async def outbound_dispatcher() -> None:
        while running:
            outbound = await bus.consume_outbound()
            await channel.send(outbound)

    agent_task = asyncio.create_task(fake_agent())
    dispatch_task = asyncio.create_task(outbound_dispatcher())

    base_url = f"http://127.0.0.1:{channel.bound_port}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            index_response = await client.get(f"{base_url}/")
            assert index_response.status_code == 200
            assert "nanobot web" in index_response.text.lower()

            health_response = await client.get(f"{base_url}/health")
            assert health_response.status_code == 200
            assert health_response.text.strip() == "ok"

        ws_url = f"ws://127.0.0.1:{channel.bound_port}/ws"
        async with websockets.connect(ws_url) as ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert hello["type"] == "hello"
            assert bool(hello.get("sid"))

            await ws.send(json.dumps({"type": "user_message", "text": "hello"}))
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert response["type"] == "assistant_message"
            assert response["text"] == "echo:hello"
    finally:
        running = False
        agent_task.cancel()
        dispatch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await agent_task
        with contextlib.suppress(asyncio.CancelledError):
            await dispatch_task
        await channel.stop()


@pytest.mark.asyncio
async def test_web_channel_security_rules() -> None:
    channel = WebChannel(
        WebConfig(enabled=True, host="0.0.0.0", port=0, token=None),
        MessageBus(),
    )
    with pytest.raises(ValueError):
        await channel.start()

    origin_channel = WebChannel(
        WebConfig(
            enabled=True,
            host="127.0.0.1",
            port=0,
            allow_origins=["https://allowed.example"],
        ),
        MessageBus(),
    )
    await origin_channel.start()
    try:
        with pytest.raises(Exception):
            await cast(Any, websockets.connect)(
                f"ws://127.0.0.1:{origin_channel.bound_port}/ws",
                origin="https://blocked.example",
            )
    finally:
        await origin_channel.stop()


@pytest.mark.asyncio
async def test_web_channel_busy_error() -> None:
    channel = WebChannel(
        WebConfig(enabled=True, host="127.0.0.1", port=0),
        MessageBus(),
    )
    await channel.start()

    try:
        uri = f"ws://127.0.0.1:{channel.bound_port}/ws?sid=busycase"
        async with websockets.connect(uri) as ws:
            _hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))

            await ws.send(json.dumps({"type": "user_message", "text": "first"}))
            await ws.send(json.dumps({"type": "user_message", "text": "second"}))

            got_busy = False
            for _ in range(10):
                payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if payload.get("type") == "error" and payload.get("code") == "busy":
                    got_busy = True
                    break
            assert got_busy
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_fixed_new_does_not_mutate_conversation_sessions(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_NoopProvider(),
        workspace=tmp_path,
    )

    fixed_key = "cli:direct"
    fixed_session = loop.sessions.get_or_create(fixed_key)
    fixed_session.add_message("user", "hello")
    fixed_session.add_message("assistant", "world")
    loop.sessions.save(fixed_session)

    conversation_key = "cli:direct"
    loop.sessions.get_or_create_for_conversation(conversation_key)
    before = loop.sessions.list_conversation_sessions(conversation_key)

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
    assert loop.sessions.get_or_create(fixed_key).messages == []

    after = loop.sessions.list_conversation_sessions(conversation_key)
    assert after["active_session_id"] == before["active_session_id"]
    assert len(after["sessions"]) == len(before["sessions"])


@pytest.mark.asyncio
async def test_fixed_session_switch_command_is_blocked(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_NoopProvider(),
        workspace=tmp_path,
    )

    fixed_key = "cli:direct"
    conversation_key = "cli:direct"
    loop.sessions.get_or_create_for_conversation(conversation_key)
    target = loop.sessions.create_session(conversation_key, title="other", switch_to=False)
    before = loop.sessions.list_conversation_sessions(conversation_key)

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

    after = loop.sessions.list_conversation_sessions(conversation_key)
    assert after["active_session_id"] == before["active_session_id"]


@pytest.mark.asyncio
async def test_web_channel_rejects_session_action_while_pending(tmp_path: Path) -> None:
    bus = MessageBus()
    session_manager = SessionManager(tmp_path)
    channel = WebChannel(
        WebConfig(enabled=True, host="127.0.0.1", port=0),
        bus,
        session_manager=session_manager,
    )
    await channel.start()

    running = True

    async def slow_agent() -> None:
        while running:
            inbound = await bus.consume_inbound()
            await asyncio.sleep(0.2)
            await bus.publish_outbound(
                OutboundMessage(
                    channel="web",
                    chat_id=inbound.chat_id,
                    content=f"echo:{inbound.content}",
                )
            )

    async def outbound_dispatcher() -> None:
        while running:
            outbound = await bus.consume_outbound()
            await channel.send(outbound)

    agent_task = asyncio.create_task(slow_agent())
    dispatch_task = asyncio.create_task(outbound_dispatcher())

    try:
        ws_url = f"ws://127.0.0.1:{channel.bound_port}/ws?sid=session-action-busy"
        async with websockets.connect(ws_url) as ws:
            _hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))

            await ws.send(json.dumps({"type": "user_message", "text": "first"}))
            await ws.send(json.dumps({"type": "session_action", "action": "list"}))

            got_busy = False
            for _ in range(12):
                payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if payload.get("type") == "error" and payload.get("code") == "busy":
                    got_busy = True
                    break
            assert got_busy
    finally:
        running = False
        agent_task.cancel()
        dispatch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await agent_task
        with contextlib.suppress(asyncio.CancelledError):
            await dispatch_task
        await channel.stop()


@pytest.mark.asyncio
async def test_web_channel_rejects_unknown_user_message_session_id(tmp_path: Path) -> None:
    bus = MessageBus()
    session_manager = SessionManager(tmp_path)
    channel = WebChannel(
        WebConfig(enabled=True, host="127.0.0.1", port=0),
        bus,
        session_manager=session_manager,
    )
    await channel.start()

    try:
        ws_url = f"ws://127.0.0.1:{channel.bound_port}/ws?sid=unknown-session-id"
        async with websockets.connect(ws_url) as ws:
            _hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))

            await ws.send(json.dumps({"type": "session_action", "action": "list"}))
            listed = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert listed["type"] == "sessions_state"
            active_id = str(listed.get("active_session_id") or "")
            assert active_id

            await ws.send(
                json.dumps(
                    {
                        "type": "user_message",
                        "text": "hello",
                        "session_id": active_id + "-stale",
                    }
                )
            )
            rejected = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert rejected["type"] == "error"
            assert rejected.get("code") == "bad_message"
            assert rejected.get("text") == "unknown session"
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_web_channel_session_actions(tmp_path: Path) -> None:
    bus = MessageBus()
    session_manager = SessionManager(tmp_path)
    channel = WebChannel(
        WebConfig(enabled=True, host="127.0.0.1", port=0),
        bus,
        session_manager=session_manager,
    )
    await channel.start()

    running = True

    async def fake_agent() -> None:
        while running:
            inbound = await bus.consume_inbound()
            await bus.publish_outbound(
                OutboundMessage(
                    channel="web",
                    chat_id=inbound.chat_id,
                    content=f"echo:{inbound.content}",
                )
            )

    async def outbound_dispatcher() -> None:
        while running:
            outbound = await bus.consume_outbound()
            await channel.send(outbound)

    agent_task = asyncio.create_task(fake_agent())
    dispatch_task = asyncio.create_task(outbound_dispatcher())

    try:
        ws_url = f"ws://127.0.0.1:{channel.bound_port}/ws?sid=session-actions"
        async with websockets.connect(ws_url) as ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert hello["type"] == "hello"

            await ws.send(json.dumps({"type": "session_action", "action": "list"}))
            listed = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert listed["type"] == "sessions_state"
            assert listed.get("active_session_id")
            assert isinstance(listed.get("sessions"), list)
            assert isinstance(listed.get("history"), list)

            await ws.send(
                json.dumps({"type": "session_action", "action": "new", "title": "Scratch"})
            )
            created = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert created["type"] == "sessions_state"
            assert any(s.get("title") == "Scratch" for s in created.get("sessions", []))
            assert isinstance(created.get("history"), list)

            active_id = str(created.get("active_session_id") or "")
            assert active_id

            await ws.send(
                json.dumps(
                    {
                        "type": "session_action",
                        "action": "pin",
                        "session_id": active_id,
                    }
                )
            )
            pinned = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert pinned["type"] == "sessions_state"
            assert any(
                s.get("id") == active_id and bool(s.get("pinned"))
                for s in pinned.get("sessions", [])
            )

            await ws.send(
                json.dumps(
                    {
                        "type": "session_action",
                        "action": "search",
                        "query": "Scratch",
                    }
                )
            )
            searched = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert searched["type"] == "sessions_state"
            assert searched.get("query") == "Scratch"
            assert all("Scratch" in str(s.get("title") or "") for s in searched.get("sessions", []))

            await ws.send(
                json.dumps(
                    {
                        "type": "user_message",
                        "text": "hello",
                        "session_id": active_id,
                    }
                )
            )
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert response["type"] == "assistant_message"
            assert response["text"] == "echo:hello"
    finally:
        running = False
        agent_task.cancel()
        dispatch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await agent_task
        with contextlib.suppress(asyncio.CancelledError):
            await dispatch_task
        await channel.stop()


@pytest.mark.asyncio
async def test_web_channel_metadata_sessions_state_omits_history(tmp_path: Path) -> None:
    session_manager = SessionManager(tmp_path)
    channel = WebChannel(
        WebConfig(enabled=True, host="127.0.0.1", port=0),
        MessageBus(),
        session_manager=session_manager,
    )

    sid = "state-light"
    conversation_key = f"web:{sid}"
    session, _ = session_manager.get_or_create_for_conversation(conversation_key)
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    session_manager.save(session)
    snapshot = session_manager.list_conversation_sessions(conversation_key)

    class DummyWS:
        def __init__(self):
            self.sent: list[dict[str, Any]] = []

        async def send(self, payload: str) -> None:
            self.sent.append(cast(dict[str, Any], json.loads(payload)))

    ws = DummyWS()
    channel._connections[sid].add(cast(Any, ws))

    await channel.send(
        OutboundMessage(
            channel="web",
            chat_id=sid,
            content="ok",
            metadata={"_nanobot": {"session_state": snapshot}},
        )
    )

    sessions_payload = next(item for item in ws.sent if item.get("type") == "sessions_state")
    assert "history" not in sessions_payload


def test_conversation_session_lifecycle(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    conversation_key = "web:abc"

    _, initial = manager.get_or_create_for_conversation(conversation_key)
    first_id = str(initial["id"])
    assert first_id
    assert initial["title"] == "New chat"

    created = manager.create_session(conversation_key, title="Build dashboard", switch_to=True)
    second_id = str(created["id"])
    assert second_id
    assert second_id != first_id

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert snapshot["active_session_id"] == second_id
    assert len(snapshot["sessions"]) == 2

    pinned = manager.set_session_pinned(conversation_key, first_id, pinned=True)
    assert pinned["pinned"] is True

    searched = manager.search_sessions(conversation_key, "initial")
    assert isinstance(searched.get("sessions"), list)

    renamed = manager.rename_session(
        conversation_key, first_id, "Initial planning", auto_title=False
    )
    assert renamed["title"] == "Initial planning"
    assert renamed["title_source"] == "manual"
    assert renamed["title_locked"] is True

    attempt = manager.note_auto_title_attempt(
        conversation_key,
        first_id,
        message_count=6,
    )
    assert attempt["title_auto_attempts"] >= 1

    snapshot = manager.list_conversation_sessions(conversation_key)
    first = next(item for item in snapshot["sessions"] if item["id"] == first_id)
    assert first["title"] == "Initial planning"
    assert first["auto_title"] is False
    assert first["pinned"] is True
    assert first["title_auto_attempts"] >= 1

    deleted = manager.delete_session(conversation_key, second_id)
    assert deleted["deleted_session_id"] == second_id
    assert deleted["recoverable"] is True

    restored = manager.restore_session(conversation_key, second_id)
    assert restored["id"] == second_id

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert any(item["id"] == second_id for item in snapshot["sessions"])

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert len(snapshot["sessions"]) == 2
    assert snapshot["active_session_id"] == second_id


def test_session_id_with_delimiter_is_rejected(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    conversation_key = "web:delimiter"

    _, initial = manager.get_or_create_for_conversation(conversation_key)
    initial_id = str(initial["id"])
    assert initial_id

    _, resolved = manager.get_or_create_for_conversation(
        conversation_key,
        requested_session_id=f"{initial_id}#bad",
    )
    assert resolved["id"] == initial_id

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert all("#" not in str(item.get("id") or "") for item in snapshot["sessions"])
    assert len(snapshot["sessions"]) == 1


def test_unknown_requested_session_id_is_rejected_without_resurrection(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    conversation_key = "web:restore-guard"

    _, initial = manager.get_or_create_for_conversation(conversation_key)
    deleted_id = str(initial["id"])
    assert deleted_id

    manager.delete_session(conversation_key, deleted_id)
    deleted_before = manager.list_deleted_sessions(conversation_key)
    assert any(item.get("id") == deleted_id for item in deleted_before.get("deleted_sessions", []))

    with pytest.raises(KeyError):
        manager.get_or_create_for_conversation(
            conversation_key,
            requested_session_id=deleted_id,
        )

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert all(item.get("id") != deleted_id for item in snapshot.get("sessions", []))
    deleted_after = manager.list_deleted_sessions(conversation_key)
    assert any(item.get("id") == deleted_id for item in deleted_after.get("deleted_sessions", []))


def test_delete_last_session_creates_replacement(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    conversation_key = "telegram:42"

    _, initial = manager.get_or_create_for_conversation(conversation_key)
    first_id = str(initial["id"])

    result = manager.delete_session(conversation_key, first_id)
    assert result["deleted_session_id"] == first_id
    assert result["created_replacement"] is True
    assert result["recoverable"] is True

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert len(snapshot["sessions"]) == 1
    assert snapshot["active_session_id"] != first_id


def test_title_policy_fields_for_auto_rename(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    conversation_key = "web:title"

    _, initial = manager.get_or_create_for_conversation(conversation_key)
    sid = str(initial["id"])

    renamed = manager.rename_session(
        conversation_key,
        sid,
        "Auto Generated Title",
        auto_title=False,
        source="auto",
        lock_title=False,
    )
    assert renamed["title_source"] == "auto"
    assert renamed["title_locked"] is False


@pytest.mark.asyncio
async def test_telegram_sessions_callback_switch_and_panel(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="dummy-token"),
        MessageBus(),
        session_manager=manager,
    )
    conversation_key = "telegram:100"

    _, first = manager.get_or_create_for_conversation(conversation_key)
    first_id = str(first["id"])
    second = manager.create_session(conversation_key, title="Second", switch_to=True)
    second_id = str(second["id"])
    assert second_id != first_id

    class DummyReplyMessage:
        def __init__(self, chat_id: int):
            self.chat_id = chat_id
            self.text = ""
            self.reply_markup = None

        async def reply_text(self, text: str, reply_markup: Any | None = None) -> None:
            self.text = text
            self.reply_markup = reply_markup

    authorized_user = SimpleNamespace(id=101, username="allowed")

    msg = DummyReplyMessage(chat_id=100)
    update = SimpleNamespace(
        message=msg,
        effective_chat=SimpleNamespace(id=100),
        effective_user=authorized_user,
    )
    await channel._on_sessions_command(cast(Any, update), cast(Any, SimpleNamespace()))
    assert "Session switcher" in msg.text
    assert msg.reply_markup is not None

    class DummyQueryMessage:
        def __init__(self, chat_id: int):
            self.chat_id = chat_id
            self.text = ""
            self.reply_markup = None

        async def edit_text(self, text: str, reply_markup: Any | None = None) -> None:
            self.text = text
            self.reply_markup = reply_markup

    class DummyQuery:
        def __init__(self, data: str, message: DummyQueryMessage, from_user: Any):
            self.data = data
            self.message = message
            self.from_user = from_user
            self.answered = False

        async def answer(self) -> None:
            self.answered = True

    query_message = DummyQueryMessage(chat_id=100)
    query = DummyQuery(f"nbs:sw:{first_id}:0", query_message, authorized_user)
    callback_update = SimpleNamespace(callback_query=query)
    await channel._on_session_callback(
        cast(Any, callback_update),
        cast(Any, SimpleNamespace()),
    )
    snapshot = manager.list_conversation_sessions(conversation_key)
    assert snapshot["active_session_id"] == first_id
    assert query.answered is True
    assert "Switched:" in query_message.text


@pytest.mark.asyncio
async def test_telegram_sessions_controls_require_allowlist(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="dummy-token", allow_from=["100"]),
        MessageBus(),
        session_manager=manager,
    )
    conversation_key = "telegram:100"

    _, first = manager.get_or_create_for_conversation(conversation_key)
    first_id = str(first["id"])
    second = manager.create_session(conversation_key, title="Second", switch_to=True)
    second_id = str(second["id"])
    assert second_id != first_id

    class DummyReplyMessage:
        def __init__(self, chat_id: int):
            self.chat_id = chat_id
            self.text = ""
            self.reply_markup = None

        async def reply_text(self, text: str, reply_markup: Any | None = None) -> None:
            self.text = text
            self.reply_markup = reply_markup

    unauthorized_user = SimpleNamespace(id=200, username="intruder")
    command_msg = DummyReplyMessage(chat_id=100)
    command_update = SimpleNamespace(
        message=command_msg,
        effective_user=unauthorized_user,
        effective_chat=SimpleNamespace(id=100),
    )

    await channel._on_sessions_command(cast(Any, command_update), cast(Any, SimpleNamespace()))
    assert command_msg.text == ""
    assert command_msg.reply_markup is None

    class DummyQueryMessage:
        def __init__(self, chat_id: int):
            self.chat_id = chat_id
            self.text = ""
            self.reply_markup = None

        async def edit_text(self, text: str, reply_markup: Any | None = None) -> None:
            self.text = text
            self.reply_markup = reply_markup

    class DummyQuery:
        def __init__(self, data: str, message: DummyQueryMessage, from_user: Any):
            self.data = data
            self.message = message
            self.from_user = from_user
            self.answered = False
            self.answer_text = None
            self.show_alert = False

        async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
            self.answered = True
            self.answer_text = text
            self.show_alert = show_alert

    query_message = DummyQueryMessage(chat_id=100)
    query = DummyQuery(f"nbs:sw:{first_id}:0", query_message, unauthorized_user)
    callback_update = SimpleNamespace(callback_query=query)
    await channel._on_session_callback(
        cast(Any, callback_update),
        cast(Any, SimpleNamespace()),
    )

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert snapshot["active_session_id"] == second_id
    assert query.answered is True
    assert query.answer_text == "You are not allowed to use this bot."
    assert query.show_alert is True
    assert query_message.text == ""


@pytest.mark.asyncio
async def test_discord_component_session_switcher(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    channel = DiscordChannel(
        DiscordConfig(enabled=True, token="dummy-token"),
        MessageBus(),
        session_manager=manager,
    )
    conversation_key = "discord:555"

    _, first = manager.get_or_create_for_conversation(conversation_key)
    first_id = str(first["id"])
    second = manager.create_session(conversation_key, title="Second", switch_to=True)
    second_id = str(second["id"])
    assert second_id != first_id

    class DummyResponse:
        def __init__(self, status_code: int = 200):
            self.status_code = status_code
            self.text = "ok"

    class DummyHTTP:
        def __init__(self):
            self.posts: list[dict[str, Any]] = []

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, Any] | None = None,
            data: dict[str, Any] | None = None,
            files: Any | None = None,
        ) -> DummyResponse:
            self.posts.append(
                {
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "data": data,
                    "files": files,
                }
            )
            return DummyResponse()

    dummy_http = DummyHTTP()
    channel._http = cast(Any, dummy_http)

    await channel._send_sessions_panel_message(channel_id="555", user_id="u1", page=0)
    assert dummy_http.posts
    first_payload = dummy_http.posts[-1]
    assert first_payload["url"].endswith("/channels/555/messages")
    assert isinstance((first_payload["json"] or {}).get("components"), list)

    interaction_payload = {
        "id": "i1",
        "token": "itok",
        "channel_id": "555",
        "member": {"user": {"id": "u1"}},
        "data": {
            "custom_id": "nbs:sel:u1:0",
            "values": [first_id],
        },
    }
    await channel._handle_interaction_create(interaction_payload)
    snapshot = manager.list_conversation_sessions(conversation_key)
    assert snapshot["active_session_id"] == first_id

    callback_payload = dummy_http.posts[-1]
    assert callback_payload["url"].endswith("/interactions/i1/itok/callback")
    body = callback_payload["json"] or {}
    assert body.get("type") == 7
    assert isinstance((body.get("data") or {}).get("components"), list)
