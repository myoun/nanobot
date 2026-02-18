import shutil
import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
import websockets
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.web import WebChannel
from nanobot.config.schema import WebConfig
from nanobot.session.manager import SessionManager

runner = CliRunner()


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

    renamed = manager.rename_session(
        conversation_key, first_id, "Initial planning", auto_title=False
    )
    assert renamed["title"] == "Initial planning"

    snapshot = manager.list_conversation_sessions(conversation_key)
    first = next(item for item in snapshot["sessions"] if item["id"] == first_id)
    assert first["title"] == "Initial planning"
    assert first["auto_title"] is False

    deleted = manager.delete_session(conversation_key, second_id)
    assert deleted["deleted_session_id"] == second_id

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert len(snapshot["sessions"]) == 1
    assert snapshot["active_session_id"] == first_id


def test_delete_last_session_creates_replacement(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    conversation_key = "telegram:42"

    _, initial = manager.get_or_create_for_conversation(conversation_key)
    first_id = str(initial["id"])

    result = manager.delete_session(conversation_key, first_id)
    assert result["deleted_session_id"] == first_id
    assert result["created_replacement"] is True

    snapshot = manager.list_conversation_sessions(conversation_key)
    assert len(snapshot["sessions"]) == 1
    assert snapshot["active_session_id"] != first_id
