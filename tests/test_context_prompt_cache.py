"""Tests for cache-friendly prompt construction."""

from __future__ import annotations

from datetime import datetime as real_datetime
from pathlib import Path
import datetime as datetime_module

from nanobot.agent.context import ContextBuilder


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 2, 24, 13, 59)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def test_system_prompt_stays_stable_when_clock_changes(tmp_path, monkeypatch) -> None:
    """System prompt should not change just because wall clock minute changes."""
    monkeypatch.setattr(datetime_module, "datetime", _FakeDatetime)

    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    _FakeDatetime.current = real_datetime(2026, 2, 24, 13, 59)
    prompt1 = builder.build_system_prompt()

    _FakeDatetime.current = real_datetime(2026, 2, 24, 14, 0)
    prompt2 = builder.build_system_prompt()

    assert prompt1 == prompt2


def test_runtime_context_is_separate_untrusted_user_message(tmp_path) -> None:
    """Runtime metadata should be a separate user message before the actual user message."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="Return exactly: OK",
        channel="cli",
        chat_id="direct",
    )

    assert messages[0]["role"] == "system"
    assert "## Current Session" not in messages[0]["content"]

    assert messages[-2]["role"] == "user"
    runtime_content = messages[-2]["content"]
    assert isinstance(runtime_content, str)
    assert ContextBuilder._RUNTIME_CONTEXT_TAG in runtime_content
    assert "Current Time:" in runtime_content
    assert "Channel: cli" in runtime_content
    assert "Chat ID: direct" in runtime_content

    assert messages[-1]["role"] == "user"
    assert "[REQUEST_ROUTING_CONTEXT]" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("Return exactly: OK")


def test_prompts_include_agent_browser_recovery_guidance(tmp_path) -> None:
    """Browser automation guidance should cover isolated Codex runtimes."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    system_prompt = builder.build_system_prompt()
    app_server_prompt = builder.build_app_server_prompt()

    assert 'Do not use skills with `available="false"`' in system_prompt
    assert "AGENT_BROWSER_EXECUTABLE_PATH" in system_prompt
    assert "AGENT_BROWSER_HOME" in system_prompt

    assert "AGENT_BROWSER_EXECUTABLE_PATH" in app_server_prompt
    assert "AGENT_BROWSER_HOME" in app_server_prompt
