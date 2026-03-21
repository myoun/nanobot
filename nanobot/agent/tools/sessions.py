"""Explicit cross-session search/read tool."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class SessionsTool(Tool):
    """Read-only tool for explicit access to other sessions/threads."""

    def __init__(self, manager: Any):
        self._manager = manager

    @property
    def name(self) -> str:
        return "sessions"

    @property
    def description(self) -> str:
        return (
            "Explicitly search and read other sessions/threads. "
            "Actions: search, read."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "read"],
                    "description": "Action to perform.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query for action=search.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum items to return.",
                    "minimum": 1,
                    "maximum": 50,
                },
                "conversation_key": {
                    "type": "string",
                    "description": "Optional conversation scope for search.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session/thread ID for action=read.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["summary", "snippet", "messages", "working_set", "transcript"],
                    "description": "Read mode for action=read.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        query: str = "",
        limit: int = 10,
        conversation_key: str | None = None,
        session_id: str | None = None,
        mode: str = "summary",
        **kwargs: Any,
    ) -> str:
        if action == "search":
            return self._search(query=query, limit=limit, conversation_key=conversation_key)
        if action == "read":
            return self._read(session_id=session_id, mode=mode, limit=limit)
        return f"Unknown action: {action}"

    def _search(self, *, query: str, limit: int, conversation_key: str | None) -> str:
        if not query.strip():
            return "Error: query is required for search"

        results = self._manager.search_sessions(
            query=query,
            limit=limit,
            conversation_key=conversation_key,
        )
        if not results:
            return "No matching sessions found."

        lines = ["Session search results:"]
        for item in results:
            session_id = str(item.get("session_id") or item.get("id") or "").strip()
            title = str(item.get("title") or "(untitled)").strip()
            conv = str(item.get("conversation_key") or "").strip()
            summary = " ".join(str(item.get("summary") or "").split())

            head = f"- {title}"
            if session_id:
                head += f" (id: {session_id})"
            if conv:
                head += f" [{conv}]"
            lines.append(head)
            if summary:
                lines.append(f"  {summary}")
        return "\n".join(lines)

    def _read(self, *, session_id: str | None, mode: str, limit: int) -> str:
        if not session_id:
            return "Error: session_id is required for read"

        result = self._manager.read_session(session_id, mode=mode, limit=limit)
        session = result.get("session", {}) if isinstance(result, dict) else {}
        title = str(session.get("title") or "(untitled)").strip()
        result_mode = str(result.get("mode") or mode) if isinstance(result, dict) else mode
        content = result.get("content") if isinstance(result, dict) else result

        header = f"Session {session_id}: {title}"
        if result_mode == "messages":
            if not isinstance(content, list) or not content:
                return header + "\nNo messages available."
            lines = [header, "Messages:"]
            for msg in content:
                role = str(msg.get("role") or "unknown").upper()
                text = " ".join(str(msg.get("content") or "").split())
                if len(text) > 200:
                    text = text[:197] + "..."
                lines.append(f"- {role}: {text or '(empty)'}")
            return "\n".join(lines)

        text = str(content or "").strip()
        if result_mode == "summary":
            label = "Summary"
        elif result_mode == "working_set":
            label = "Working Set"
        elif result_mode == "transcript":
            label = "Transcript"
        else:
            label = "Snippet"
        if not text:
            text = "(empty)"
        return f"{header}\n{label}:\n{text}"
