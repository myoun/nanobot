from typing import Any

import pytest

from nanobot.agent.tools.sessions import SessionsTool


class FakeManager:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []

    def search_sessions(
        self,
        query: str,
        *,
        limit: int = 10,
        conversation_key: str | None = None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append(
            {
                "query": query,
                "limit": limit,
                "conversation_key": conversation_key,
            }
        )
        return [
            {
                "session_id": "sess_1",
                "conversation_key": "web:chat-1",
                "title": "DB migration",
                "summary": "Discussed the backfill and rollback plan.",
            }
        ]

    def read_session(
        self,
        session_id: str,
        *,
        mode: str = "summary",
        limit: int = 50,
    ) -> dict[str, Any]:
        self.read_calls.append(
            {
                "session_id": session_id,
                "mode": mode,
                "limit": limit,
            }
        )
        if mode == "messages":
            content: Any = [
                {"role": "user", "content": "Find the last migration thread."},
                {"role": "assistant", "content": "Found the session and summarized it."},
            ]
        else:
            content = "Discussed the backfill and rollback plan."
        return {
            "session": {
                "id": session_id,
                "title": "DB migration",
            },
            "mode": mode,
            "content": content,
        }


@pytest.mark.asyncio
async def test_sessions_tool_search_formats_hits() -> None:
    manager = FakeManager()
    tool = SessionsTool(manager)

    result = await tool.execute(
        action="search",
        query="migration",
        limit=5,
        conversation_key="web:chat-1",
    )

    assert "Session search results:" in result
    assert "DB migration" in result
    assert "sess_1" in result
    assert "[web:chat-1]" in result
    assert manager.search_calls == [
        {
            "query": "migration",
            "limit": 5,
            "conversation_key": "web:chat-1",
        }
    ]


@pytest.mark.asyncio
async def test_sessions_tool_search_requires_query() -> None:
    tool = SessionsTool(FakeManager())

    result = await tool.execute(action="search", query="")

    assert result == "Error: query is required for search"


@pytest.mark.asyncio
async def test_sessions_tool_read_formats_messages() -> None:
    manager = FakeManager()
    tool = SessionsTool(manager)

    result = await tool.execute(action="read", session_id="sess_1", mode="messages", limit=2)

    assert "Session sess_1: DB migration" in result
    assert "- USER: Find the last migration thread." in result
    assert "- ASSISTANT: Found the session and summarized it." in result
    assert manager.read_calls == [
        {
            "session_id": "sess_1",
            "mode": "messages",
            "limit": 2,
        }
    ]


@pytest.mark.asyncio
async def test_sessions_tool_read_requires_session_id() -> None:
    tool = SessionsTool(FakeManager())

    result = await tool.execute(action="read", session_id=None)

    assert result == "Error: session_id is required for read"
