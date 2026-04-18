"""Tests for current MemoryStore consolidation behavior."""

import json
from pathlib import Path

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.manager import Session


def _make_session(message_count: int = 30) -> Session:
    session = Session(key="test:memory")
    for i in range(message_count):
        session.messages.append(
            {
                "role": "user",
                "content": f"msg{i}",
                "timestamp": "2026-01-01 00:00",
            }
        )
    return session


def _make_tool_response(history_entry, memory_update) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="save_memory",
                arguments={
                    "history_entry": history_entry,
                    "memory_update": memory_update,
                },
            )
        ],
    )


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


@pytest.mark.asyncio
async def test_string_arguments_work(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    provider = ScriptedProvider(
        [
            _make_tool_response(
                history_entry="[2026-01-01] User discussed testing.",
                memory_update="# Memory\nUser likes testing.",
            )
        ]
    )
    session = _make_session(message_count=60)

    result = await store.consolidate(session, provider, "test-model")

    assert result is True
    assert "[2026-01-01] User discussed testing." in store.history_file.read_text()
    assert "User likes testing." in store.memory_file.read_text()
    assert session.last_consolidated == len(session.messages) - 25


@pytest.mark.asyncio
async def test_dict_arguments_are_serialized(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    provider = ScriptedProvider(
        [
            _make_tool_response(
                history_entry={"timestamp": "2026-01-01", "summary": "User discussed testing."},
                memory_update={"facts": ["User likes testing"]},
            )
        ]
    )
    session = _make_session(message_count=60)

    result = await store.consolidate(session, provider, "test-model")

    assert result is True
    history_entry = json.loads(store.history_file.read_text().strip())
    assert history_entry["summary"] == "User discussed testing."
    memory_doc = json.loads(store.memory_file.read_text())
    assert memory_doc["facts"] == ["User likes testing"]


@pytest.mark.asyncio
async def test_json_string_arguments_are_parsed(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="save_memory",
                        arguments=json.dumps(
                            {
                                "history_entry": "[2026-01-01] User discussed testing.",
                                "memory_update": "# Memory\nUser likes testing.",
                            }
                        ),
                    )
                ],
            )
        ]
    )
    session = _make_session(message_count=60)

    result = await store.consolidate(session, provider, "test-model")

    assert result is True
    assert "User discussed testing." in store.history_file.read_text()


@pytest.mark.asyncio
async def test_no_tool_call_returns_false(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    provider = ScriptedProvider([LLMResponse(content="summary", tool_calls=[])])
    session = _make_session(message_count=60)

    result = await store.consolidate(session, provider, "test-model")

    assert result is False
    assert not store.history_file.exists()


@pytest.mark.asyncio
async def test_empty_or_small_session_is_noop(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    provider = ScriptedProvider([])
    session = _make_session(message_count=10)

    result = await store.consolidate(session, provider, "test-model")

    assert result is True
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_archive_all_uses_entire_session(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    provider = ScriptedProvider(
        [
            _make_tool_response(
                history_entry="[2026-01-01] Archived everything.",
                memory_update="# Memory\nAll archived.",
            )
        ]
    )
    session = _make_session(message_count=5)

    result = await store.consolidate(
        session,
        provider,
        "test-model",
        archive_all=True,
    )

    assert result is True
    assert session.last_consolidated == 0
    assert "Archived everything." in store.history_file.read_text()
