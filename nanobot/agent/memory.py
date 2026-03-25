"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import weakref
from datetime import datetime
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import (
    ensure_dir,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    get_data_path,
    safe_filename,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        data_root = get_data_path()
        self.memory_dir = ensure_dir(data_root / "memories" / "_legacy")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.item_root = ensure_dir(data_root / "memories")
        try:
            resolved_name = workspace.resolve().name  # type: ignore[union-attr]
        except Exception:
            resolved_name = None
        if not isinstance(resolved_name, str) or not resolved_name.strip():
            fallback_name = getattr(workspace, "name", None)
            resolved_name = fallback_name if isinstance(fallback_name, str) else "workspace"
        self.workspace_id = safe_filename(resolved_name or "workspace")
        self.global_instructions_dir = ensure_dir(self.item_root / "global" / "instructions")
        self.global_facts_dir = ensure_dir(self.item_root / "global" / "facts")
        self.global_preferences_dir = ensure_dir(self.item_root / "global" / "preferences")
        self.workspace_rules_dir = ensure_dir(self.item_root / "workspaces" / self.workspace_id / "rules")
        self.workspace_memory_dir = ensure_dir(self.item_root / "workspaces" / self.workspace_id / "memory")

    def list_items(
        self,
        *,
        scope: str | None = None,
        kind: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.search_items("", limit=limit, scope=scope, kind=kind)

    def create_item(
        self,
        *,
        scope: str,
        kind: str,
        title: str,
        content: str,
        slug: str | None = None,
    ) -> dict[str, Any]:
        directory = self._directory_for(scope=scope, kind=kind)
        item_path = self._allocate_item_path(directory, slug=slug or title)
        rendered = self._render_item_markdown(title=title, content=content)
        item_path.write_text(rendered, encoding="utf-8")
        self.rebuild_index()
        item_id = item_path.relative_to(self.item_root).as_posix()
        return {
            "item_id": item_id,
            "scope": scope,
            "kind": kind,
            "title": title.strip(),
            "path": str(item_path),
            "updated_at": datetime.fromtimestamp(item_path.stat().st_mtime).isoformat(),
        }

    def update_item(
        self,
        *,
        item_id: str,
        content: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        item_path = self._path_for_item_id(item_id)
        existing_title = title or self._title_from_item_file(item_path)
        rendered = self._render_item_markdown(title=existing_title, content=content)
        item_path.write_text(rendered, encoding="utf-8")
        self.rebuild_index()

        parts = Path(item_id).parts
        scope = parts[0] if parts else ""
        kind = parts[1] if len(parts) >= 2 else "memory"
        if scope == "workspaces" and len(parts) >= 3:
            scope = f"workspace:{parts[1]}"
            kind = parts[2]
        return {
            "item_id": item_id,
            "scope": scope,
            "kind": kind,
            "title": existing_title.strip(),
            "path": str(item_path),
            "updated_at": datetime.fromtimestamp(item_path.stat().st_mtime).isoformat(),
        }

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        sections: list[str] = []

        if global_instructions := self._read_item_group(self.global_instructions_dir):
            sections.append(f"## Global Instructions\n{global_instructions}")

        if workspace_rules := self._read_item_group(self.workspace_rules_dir):
            sections.append(f"## Workspace Rules\n{workspace_rules}")

        if workspace_memory := self._read_item_group(self.workspace_memory_dir):
            sections.append(f"## Workspace Memory\n{workspace_memory}")

        global_memory_parts = [
            self._read_item_group(self.global_facts_dir),
            self._read_item_group(self.global_preferences_dir),
        ]
        global_memory = "\n\n".join(part for part in global_memory_parts if part)
        if global_memory:
            sections.append(f"## Global Memory\n{global_memory}")

        if long_term := self.read_long_term():
            sections.append(f"## Legacy Workspace Memory\n{long_term}")

        return "\n\n".join(section for section in sections if section)

    def rebuild_index(self) -> int:
        from nanobot.session.search_index import SessionArtifactIndex

        return SessionArtifactIndex(self.workspace).rebuild_memory_index()

    def search_items(
        self,
        query: str,
        *,
        limit: int = 10,
        scope: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        from nanobot.session.search_index import SessionArtifactIndex

        return SessionArtifactIndex(self.workspace).search_memory(
            query=query,
            limit=limit,
            scope=scope,
            kind=kind,
        )

    def read_item(self, item_id: str) -> dict[str, Any] | None:
        from nanobot.session.search_index import SessionArtifactIndex

        return SessionArtifactIndex(self.workspace).read_memory_item(item_id)

    def _directory_for(self, *, scope: str, kind: str) -> Path:
        normalized_scope = scope.strip().lower()
        normalized_kind = kind.strip().lower()
        if normalized_scope == "global":
            mapping = {
                "instructions": self.global_instructions_dir,
                "facts": self.global_facts_dir,
                "preferences": self.global_preferences_dir,
            }
        elif normalized_scope == "workspace":
            mapping = {
                "rules": self.workspace_rules_dir,
                "memory": self.workspace_memory_dir,
            }
        else:
            raise ValueError("scope must be 'global' or 'workspace'")

        directory = mapping.get(normalized_kind)
        if directory is None:
            allowed = ", ".join(sorted(mapping))
            raise ValueError(f"kind must be one of: {allowed}")
        return directory

    def _path_for_item_id(self, item_id: str) -> Path:
        raw = item_id.strip().strip("/")
        if not raw:
            raise ValueError("item_id is required")
        path = (self.item_root / raw).resolve()
        item_root_resolved = self.item_root.resolve()
        if item_root_resolved not in path.parents and path != item_root_resolved:
            raise ValueError("item_id must stay inside the memories root")
        if not path.exists():
            raise ValueError(f"Unknown memory item: {item_id}")
        if not path.is_file():
            raise ValueError(f"Memory item is not a file: {item_id}")
        return path

    @staticmethod
    def _slugify(value: str) -> str:
        base = safe_filename(value.strip().lower() or "memory")
        base = "-".join(part for part in base.replace("_", " ").split() if part)
        return base or "memory"

    def _allocate_item_path(self, directory: Path, *, slug: str) -> Path:
        base = self._slugify(slug)
        candidate = directory / f"{base}.md"
        suffix = 2
        while candidate.exists():
            candidate = directory / f"{base}-{suffix}.md"
            suffix += 1
        return candidate

    @staticmethod
    def _render_item_markdown(*, title: str, content: str) -> str:
        heading = title.strip() or "Memory"
        body = content.strip()
        if not body:
            raise ValueError("content must not be empty")
        return f"# {heading}\n\n{body}\n"

    @staticmethod
    def _title_from_item_file(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.stem
        return path.stem.replace("-", " ").replace("_", " ").strip() or path.stem

    def _read_item_group(self, directory: Path) -> str:
        if not directory.exists():
            return ""

        entries: list[str] = []
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            entries.append(text)
        return "\n\n---\n\n".join(entries)

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning("Memory consolidation: unexpected arguments type {}", type(args).__name__)
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False


class MemoryConsolidator:
    """Compatibility wrapper for token-based memory consolidation."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3
    _SAFETY_BUFFER = 1024

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages,
        get_tool_definitions,
        max_completion_tokens: int = 4096,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        from nanobot.session.manager import Session

        if not messages:
            return True
        snapshot = Session(key="memory:archive")
        snapshot.messages = [dict(message) for message in messages]
        return await self.store.consolidate(
            snapshot,
            self.provider,
            self.model,
            archive_all=True,
        )

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    @staticmethod
    def _format_messages(messages: list[dict[str, object]]) -> str:
        lines: list[str] = []
        for message in messages:
            content = message.get("content")
            if not content:
                continue
            tools_used = message.get("tools_used")
            tools = (
                f" [tools: {', '.join(tools_used)}]"
                if isinstance(tools_used, list) and tools_used
                else ""
            )
            timestamp = str(message.get("timestamp", "?"))[:16]
            role = str(message.get("role", "assistant")).upper()
            lines.append(f"[{timestamp}] {role}{tools}: {content}")
        return "\n".join(lines)

    def _raw_archive(self, messages: list[dict[str, object]]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.store.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )

    async def archive_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive messages with guaranteed persistence."""
        if not messages:
            return True
        for _ in range(self._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(messages):
                return True
        self._raw_archive(messages)
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Archive old messages until prompt fits within the safe token budget."""
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < budget:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
