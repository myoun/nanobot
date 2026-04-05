"""Tests for /restart slash command."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestRestartCommand:

    @pytest.mark.asyncio
    async def test_restart_requests_graceful_shutdown(self):
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        out = await cmd_restart(ctx)

        assert "Restart requested" in out.content
        assert loop.restart_requested is True

    @pytest.mark.asyncio
    async def test_restart_intercepted_in_run_loop(self):
        """Verify /restart is handled at the run-loop level, not inside _dispatch."""
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/restart")

        with patch.object(loop, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            await bus.publish_inbound(msg)

            run_task = asyncio.create_task(loop.run())
            await asyncio.wait_for(run_task, timeout=1.0)

            mock_dispatch.assert_not_called()
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "Restart requested" in out.content
            assert loop.restart_requested is True

    @pytest.mark.asyncio
    async def test_status_intercepted_in_run_loop(self):
        """Verify non-priority slash commands are routed through _dispatch."""
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")

        with patch.object(loop, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(loop.run())
            await asyncio.sleep(0.1)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_propagates_external_cancellation(self):
        """External task cancellation should not be swallowed by the inbound wait loop."""
        loop, _bus = _make_loop()

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.1)
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_help_includes_restart(self):
        loop, _bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/help")

        response = await loop._process_message(msg)

        assert response is not None
        assert "/new" in response.content
        assert "/status" in response.content
        assert response.metadata == {}

    @pytest.mark.asyncio
    async def test_status_reports_runtime_info(self):
        loop, _bus = _make_loop()
        session = Session(key="telegram:c1", metadata={})
        loop.sessions.get_active_session.return_value = session
        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")

        response = await loop._process_message(msg)

        assert response is not None
        assert "Model: test-model" in response.content
        assert "Routing: enabled" in response.content
        assert "Tool hints: enabled" in response.content
        assert "5h limit: unavailable" in response.content
        assert "Weekly limit: unavailable" in response.content
        assert "Context window: unavailable" in response.content
        assert response.metadata == {}

    @pytest.mark.asyncio
    async def test_run_agent_loop_resets_usage_when_provider_omits_it(self):
        loop, _bus = _make_loop()
        loop.provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(content="first", usage={"prompt_tokens": 9, "completion_tokens": 4}),
            LLMResponse(content="second", usage={}),
        ])

        await loop._run_agent_loop([])
        assert loop._last_usage == {"prompt_tokens": 9, "completion_tokens": 4}

        await loop._run_agent_loop([])
        assert loop._last_usage == {"prompt_tokens": 0, "completion_tokens": 0}

    @pytest.mark.asyncio
    async def test_status_falls_back_to_last_usage_when_context_estimate_missing(self):
        loop, _bus = _make_loop()
        session = Session(key="telegram:c1", metadata={})
        loop.sessions.get_active_session.return_value = session
        loop.sessions.get_or_create.return_value = session
        session.metadata["app_server_token_usage"] = {"total": {"totalTokens": 1200}}
        loop.provider.get_runtime_status = AsyncMock(
            return_value={"config": {"model_context_window": 4000}}
        )

        response = await loop._process_message(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")
        )

        assert response is not None
        assert "Context window: last turn used ~1,200 tokens" in response.content

    @pytest.mark.asyncio
    async def test_process_direct_preserves_render_metadata(self):
        loop, _bus = _make_loop()
        session = Session(key="cli:test", metadata={})
        loop.sessions.get_or_create.return_value = session

        response = await loop.process_direct("/status", session_key="cli:test")

        assert response is not None
        assert response.metadata == {}
