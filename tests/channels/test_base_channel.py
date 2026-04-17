from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.web import WebChannel
from nanobot.config.schema import WebConfig


class _DummyChannel(BaseChannel):
    name = "dummy"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, msg: OutboundMessage) -> None:
        return None


def test_is_allowed_requires_exact_match() -> None:
    channel = _DummyChannel(SimpleNamespace(allow_from=["allow@email.com"]), MessageBus())

    assert channel.is_allowed("allow@email.com") is True
    assert channel.is_allowed("attacker|allow@email.com") is False


class _DummyWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send(self, raw: str) -> None:
        self.messages.append(json.loads(raw))


@pytest.mark.asyncio
async def test_web_channel_send_keeps_pending_for_concurrent_command_response() -> None:
    channel = WebChannel(WebConfig(), MessageBus())
    websocket = _DummyWebSocket()
    sid = "session-1"
    channel._connections[sid].add(websocket)
    channel._pending.add(sid)

    await channel.send(
        OutboundMessage(
            channel="web",
            chat_id=sid,
            content="status snapshot",
            metadata={"_web_blocking_request": False},
        )
    )

    assert sid in channel._pending
    assert websocket.messages[-1]["clearsBusy"] is False


@pytest.mark.asyncio
async def test_web_channel_send_clears_pending_for_blocking_response() -> None:
    channel = WebChannel(WebConfig(), MessageBus())
    websocket = _DummyWebSocket()
    sid = "session-1"
    channel._connections[sid].add(websocket)
    channel._pending.add(sid)

    await channel.send(
        OutboundMessage(
            channel="web",
            chat_id=sid,
            content="final answer",
            metadata={"_web_blocking_request": True},
        )
    )

    assert sid not in channel._pending
    assert websocket.messages[-1]["clearsBusy"] is True
