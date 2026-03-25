"""Explicit long-term memory tool."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class MemoryTool(Tool):
    """Read/write tool for explicit global and workspace memories."""

    def __init__(self, store: Any):
        self._store = store

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return (
            "Create, list, search, read, and update durable memory items. "
            "Use this for global preferences/facts/instructions and workspace rules/memory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "search", "read", "update"],
                },
                "scope": {
                    "type": "string",
                    "enum": ["global", "workspace"],
                    "description": "Scope for create/list/search.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["instructions", "facts", "preferences", "rules", "memory"],
                    "description": "Kind of memory item.",
                },
                "title": {
                    "type": "string",
                    "description": "Short title for create or update.",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown body for create or update.",
                },
                "slug": {
                    "type": "string",
                    "description": "Optional filename stem for create.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query for action=search.",
                },
                "item_id": {
                    "type": "string",
                    "description": "Memory item id for read or update.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum items to return for list/search.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        scope: str | None = None,
        kind: str | None = None,
        title: str = "",
        content: str = "",
        slug: str | None = None,
        query: str = "",
        item_id: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> str:
        if action == "create":
            return self._create(scope=scope, kind=kind, title=title, content=content, slug=slug)
        if action == "list":
            return self._list(scope=scope, kind=kind, limit=limit)
        if action == "search":
            return self._search(query=query, scope=scope, kind=kind, limit=limit)
        if action == "read":
            return self._read(item_id=item_id)
        if action == "update":
            return self._update(item_id=item_id, title=title, content=content)
        return f"Unknown action: {action}"

    def _create(
        self,
        *,
        scope: str | None,
        kind: str | None,
        title: str,
        content: str,
        slug: str | None,
    ) -> str:
        if not scope:
            return "Error: scope is required for create"
        if not kind:
            return "Error: kind is required for create"
        if not title.strip():
            return "Error: title is required for create"
        if not content.strip():
            return "Error: content is required for create"
        record = self._store.create_item(
            scope=scope,
            kind=kind,
            title=title,
            content=content,
            slug=slug,
        )
        return (
            "Memory saved.\n"
            f"- item_id: {record['item_id']}\n"
            f"- scope: {record['scope']}\n"
            f"- kind: {record['kind']}\n"
            f"- path: {record['path']}"
        )

    def _list(self, *, scope: str | None, kind: str | None, limit: int) -> str:
        items = self._store.list_items(scope=scope, kind=kind, limit=limit)
        if not items:
            return "No memory items found."
        lines = ["Memory items:"]
        for item in items:
            title = str(item.get("title") or item.get("item_id") or "(untitled)").strip()
            lines.append(
                f"- {title} (item_id: {item.get('item_id')}, scope: {item.get('scope')}, kind: {item.get('kind')})"
            )
        return "\n".join(lines)

    def _search(self, *, query: str, scope: str | None, kind: str | None, limit: int) -> str:
        if not query.strip():
            return "Error: query is required for search"
        items = self._store.search_items(query=query, scope=scope, kind=kind, limit=limit)
        if not items:
            return "No matching memory items found."
        lines = ["Memory search results:"]
        for item in items:
            title = str(item.get("title") or item.get("item_id") or "(untitled)").strip()
            lines.append(
                f"- {title} (item_id: {item.get('item_id')}, scope: {item.get('scope')}, kind: {item.get('kind')})"
            )
            snippet = " ".join(str(item.get("snippet") or "").split())
            if snippet:
                lines.append(f"  {snippet}")
        return "\n".join(lines)

    def _read(self, *, item_id: str | None) -> str:
        if not item_id:
            return "Error: item_id is required for read"
        result = self._store.read_item(item_id)
        if result is None:
            return f"Memory item not found: {item_id}"
        item = result.get("item", {})
        content = str(result.get("content") or "").strip() or "(empty)"
        return (
            f"Memory {item_id}\n"
            f"Title: {item.get('title')}\n"
            f"Scope: {item.get('scope')}\n"
            f"Kind: {item.get('kind')}\n"
            f"Path: {item.get('path')}\n\n"
            f"{content}"
        )

    def _update(self, *, item_id: str | None, title: str, content: str) -> str:
        if not item_id:
            return "Error: item_id is required for update"
        if not content.strip():
            return "Error: content is required for update"
        record = self._store.update_item(item_id=item_id, title=title or None, content=content)
        return (
            "Memory updated.\n"
            f"- item_id: {record['item_id']}\n"
            f"- title: {record['title']}\n"
            f"- path: {record['path']}"
        )
