"""Session management for conversation history."""

from __future__ import annotations

import json
import secrets
import string
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Get recent messages in LLM format, preserving tool metadata."""
        out: list[dict[str, Any]] = []
        for m in self.messages[-max_messages:]:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    Multi-session metadata is stored in `_session_index.json`.
    """

    _INDEX_FILENAME = "_session_index.json"
    _DEFAULT_TITLE = "New chat"
    _SESSION_ID_ALPHABET = string.ascii_lowercase + string.digits

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = Path.home() / ".nanobot" / "sessions"
        self._cache: dict[str, Session] = {}
        self._index_cache: dict[str, Any] | None = None

    @staticmethod
    def compose_key(conversation_key: str, session_id: str) -> str:
        """Build full storage key for a session under a conversation."""
        return f"{conversation_key}#{session_id}"

    @staticmethod
    def split_composed_key(key: str) -> tuple[str, str] | None:
        """Split `channel:chat_id#session_id` into `(conversation_key, session_id)`."""
        if "#" not in key:
            return None
        conversation_key, session_id = key.rsplit("#", 1)
        if not conversation_key or not session_id:
            return None
        return conversation_key, session_id

    @staticmethod
    def _iso_now() -> str:
        return datetime.now().isoformat()

    @classmethod
    def _new_session_id(cls, length: int = 10) -> str:
        safe_len = max(6, min(32, int(length)))
        return "".join(secrets.choice(cls._SESSION_ID_ALPHABET) for _ in range(safe_len))

    @classmethod
    def _normalize_title(cls, title: str | None, fallback: str | None = None) -> str:
        raw = (title or "").strip()
        if not raw:
            raw = fallback or cls._DEFAULT_TITLE
        compact = " ".join(raw.split())
        return compact[:80]

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    @property
    def _index_path(self) -> Path:
        return self.sessions_dir / self._INDEX_FILENAME

    def _empty_index(self) -> dict[str, Any]:
        return {"version": 1, "conversations": {}}

    def _load_index(self) -> dict[str, Any]:
        if self._index_cache is not None:
            return self._index_cache

        if not self._index_path.exists():
            self._index_cache = self._empty_index()
            return self._index_cache

        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load session index, recreating: {e}")
            data = self._empty_index()

        if not isinstance(data, dict):
            data = self._empty_index()
        if not isinstance(data.get("conversations"), dict):
            data["conversations"] = {}
        if not isinstance(data.get("version"), int):
            data["version"] = 1

        self._index_cache = data
        return data

    def _save_index(self, index: dict[str, Any]) -> None:
        self._index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._index_cache = index

    def _find_meta(self, entry: dict[str, Any], session_id: str) -> dict[str, Any] | None:
        sessions = entry.get("sessions")
        if not isinstance(sessions, list):
            return None
        for item in sessions:
            if isinstance(item, dict) and str(item.get("id", "")) == session_id:
                return item
        return None

    def _append_meta(
        self,
        entry: dict[str, Any],
        conversation_key: str,
        session_id: str,
        title: str | None,
        *,
        auto_title: bool,
    ) -> dict[str, Any]:
        now = self._iso_now()
        meta = {
            "id": session_id,
            "key": self.compose_key(conversation_key, session_id),
            "title": self._normalize_title(title),
            "created_at": now,
            "updated_at": now,
            "auto_title": bool(auto_title),
        }
        sessions = entry.get("sessions")
        if not isinstance(sessions, list):
            sessions = []
            entry["sessions"] = sessions
        sessions.append(meta)
        return meta

    def _bootstrap_conversation(self, conversation_key: str) -> tuple[dict[str, Any], bool]:
        """
        Ensure conversation exists in index.

        Returns `(entry, changed)`.
        """
        index = self._load_index()
        conversations = index.get("conversations")
        if not isinstance(conversations, dict):
            conversations = {}
            index["conversations"] = conversations

        existing = conversations.get(conversation_key)
        if (
            isinstance(existing, dict)
            and isinstance(existing.get("sessions"), list)
            and existing.get("sessions")
        ):
            if not existing.get("active_session_id"):
                first = existing["sessions"][0]
                if isinstance(first, dict) and first.get("id"):
                    existing["active_session_id"] = str(first["id"])
                    return existing, True
            return existing, False

        entry: dict[str, Any] = {"active_session_id": "", "sessions": []}
        conversations[conversation_key] = entry

        legacy_current = self._get_session_path(conversation_key)
        if legacy_current.exists():
            sid = self._new_session_id()
            migrated_key = self.compose_key(conversation_key, sid)
            migrated_path = self._get_session_path(migrated_key)
            try:
                legacy_current.rename(migrated_path)
                logger.info(
                    f"Migrated conversation session to keyed format: {conversation_key} -> {migrated_key}"
                )
            except Exception as e:
                logger.warning(f"Failed to migrate legacy session file {legacy_current}: {e}")
                sid = self._new_session_id()
            meta = self._append_meta(
                entry, conversation_key, sid, title="Migrated chat", auto_title=False
            )
            entry["active_session_id"] = meta["id"]
            return entry, True

        sid = self._new_session_id()
        meta = self._append_meta(
            entry, conversation_key, sid, title=self._DEFAULT_TITLE, auto_title=True
        )
        entry["active_session_id"] = meta["id"]
        return entry, True

    def _ensure_conversation(self, conversation_key: str) -> dict[str, Any]:
        entry, changed = self._bootstrap_conversation(conversation_key)
        if changed:
            self._save_index(self._load_index())
        return entry

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key.

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def get_or_create_for_conversation(
        self,
        conversation_key: str,
        requested_session_id: str | None = None,
    ) -> tuple[Session, dict[str, Any]]:
        """
        Resolve the active session for a conversation, optionally switching first.

        Returns `(session, descriptor)` where descriptor includes id/title/active.
        """
        entry = self._ensure_conversation(conversation_key)
        index = self._load_index()
        changed = False

        target_id = (requested_session_id or "").strip()
        if target_id:
            if self._find_meta(entry, target_id) is None:
                self._append_meta(
                    entry,
                    conversation_key,
                    target_id,
                    title=self._DEFAULT_TITLE,
                    auto_title=True,
                )
                changed = True
            if entry.get("active_session_id") != target_id:
                entry["active_session_id"] = target_id
                changed = True

        active_id = str(entry.get("active_session_id") or "")
        if not active_id:
            first = next((s for s in entry.get("sessions", []) if isinstance(s, dict)), None)
            if first and first.get("id"):
                active_id = str(first["id"])
                entry["active_session_id"] = active_id
                changed = True

        if not active_id:
            sid = self._new_session_id()
            meta = self._append_meta(
                entry, conversation_key, sid, title=self._DEFAULT_TITLE, auto_title=True
            )
            active_id = str(meta["id"])
            entry["active_session_id"] = active_id
            changed = True

        meta = self._find_meta(entry, active_id)
        if meta is None:
            meta = self._append_meta(
                entry, conversation_key, active_id, title=self._DEFAULT_TITLE, auto_title=True
            )
            changed = True

        if changed:
            self._save_index(index)

        key = str(meta.get("key") or self.compose_key(conversation_key, active_id))
        session = self.get_or_create(key)

        title = self._normalize_title(meta.get("title"), self._DEFAULT_TITLE)
        if session.metadata.get("title") != title:
            session.metadata["title"] = title

        descriptor = {
            "id": active_id,
            "key": key,
            "title": title,
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "auto_title": bool(meta.get("auto_title", True)),
            "active": True,
        }
        return session, descriptor

    def list_conversation_sessions(self, conversation_key: str) -> dict[str, Any]:
        """List all sessions for a given conversation key."""
        entry = self._ensure_conversation(conversation_key)
        active_id = str(entry.get("active_session_id") or "")
        out: list[dict[str, Any]] = []
        for item in entry.get("sessions", []):
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or "")
            if not sid:
                continue
            out.append(
                {
                    "id": sid,
                    "key": str(item.get("key") or self.compose_key(conversation_key, sid)),
                    "title": self._normalize_title(item.get("title"), self._DEFAULT_TITLE),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "auto_title": bool(item.get("auto_title", True)),
                    "active": sid == active_id,
                }
            )
        out.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
        return {
            "conversation_key": conversation_key,
            "active_session_id": active_id,
            "sessions": out,
        }

    def create_session(
        self,
        conversation_key: str,
        title: str | None = None,
        *,
        switch_to: bool = True,
    ) -> dict[str, Any]:
        """Create a session under a conversation and optionally switch to it."""
        index = self._load_index()
        conversations = index.get("conversations")
        if not isinstance(conversations, dict):
            conversations = {}
            index["conversations"] = conversations

        entry = conversations.get(conversation_key)
        if not isinstance(entry, dict):
            if self._get_session_path(conversation_key).exists():
                entry = self._ensure_conversation(conversation_key)
            else:
                entry = {"active_session_id": "", "sessions": []}
                conversations[conversation_key] = entry

        sessions = entry.get("sessions")
        if not isinstance(sessions, list):
            sessions = []
            entry["sessions"] = sessions

        existing_ids = {
            str(item.get("id")) for item in sessions if isinstance(item, dict) and item.get("id")
        }
        sid = self._new_session_id()
        while sid in existing_ids:
            sid = self._new_session_id()

        auto_title = not bool((title or "").strip())
        meta = self._append_meta(entry, conversation_key, sid, title=title, auto_title=auto_title)
        if switch_to:
            entry["active_session_id"] = sid
        self._save_index(index)

        session = Session(
            key=str(meta["key"]),
            metadata={"title": self._normalize_title(meta.get("title"), self._DEFAULT_TITLE)},
        )
        self.save(session)

        result = {
            "id": sid,
            "key": str(meta["key"]),
            "title": self._normalize_title(meta.get("title"), self._DEFAULT_TITLE),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "auto_title": bool(meta.get("auto_title", True)),
            "active": bool(switch_to),
        }
        return result

    def switch_session(self, conversation_key: str, session_id: str) -> dict[str, Any]:
        """Switch active session in a conversation."""
        entry = self._ensure_conversation(conversation_key)
        index = self._load_index()
        sid = (session_id or "").strip()
        meta = self._find_meta(entry, sid)
        if meta is None:
            raise KeyError(f"Session not found: {sid}")
        entry["active_session_id"] = sid
        self._save_index(index)
        return {
            "id": sid,
            "key": str(meta.get("key") or self.compose_key(conversation_key, sid)),
            "title": self._normalize_title(meta.get("title"), self._DEFAULT_TITLE),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "auto_title": bool(meta.get("auto_title", True)),
            "active": True,
        }

    def rename_session(
        self,
        conversation_key: str,
        session_id: str,
        new_title: str,
        *,
        auto_title: bool = False,
    ) -> dict[str, Any]:
        """Rename a session."""
        entry = self._ensure_conversation(conversation_key)
        index = self._load_index()
        sid = (session_id or "").strip()
        meta = self._find_meta(entry, sid)
        if meta is None:
            raise KeyError(f"Session not found: {sid}")

        normalized = self._normalize_title(new_title)
        meta["title"] = normalized
        meta["auto_title"] = bool(auto_title)
        meta["updated_at"] = self._iso_now()
        self._save_index(index)

        key = str(meta.get("key") or self.compose_key(conversation_key, sid))
        session = self.get_or_create(key)
        session.metadata["title"] = normalized
        self.save(session)

        return {
            "id": sid,
            "key": key,
            "title": normalized,
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "auto_title": bool(meta.get("auto_title", False)),
            "active": str(entry.get("active_session_id", "")) == sid,
        }

    def delete_session(self, conversation_key: str, session_id: str) -> dict[str, Any]:
        """
        Delete a session and switch to another one.

        If this was the last session, a replacement session is created automatically.
        """
        entry = self._ensure_conversation(conversation_key)
        index = self._load_index()
        sid = (session_id or "").strip()
        sessions = [s for s in entry.get("sessions", []) if isinstance(s, dict)]
        target = self._find_meta(entry, sid)
        if target is None:
            raise KeyError(f"Session not found: {sid}")

        key = str(target.get("key") or self.compose_key(conversation_key, sid))
        entry["sessions"] = [s for s in sessions if str(s.get("id", "")) != sid]
        self.invalidate(key)
        try:
            self._get_session_path(key).unlink(missing_ok=True)
        except Exception:
            pass

        created_replacement = False
        active_id = str(entry.get("active_session_id") or "")
        remaining = [s for s in entry.get("sessions", []) if isinstance(s, dict)]
        if not remaining:
            replacement = self.create_session(conversation_key, title=None, switch_to=True)
            created_replacement = True
            active_id = replacement["id"]
            remaining_snapshot = self.list_conversation_sessions(conversation_key)
            return {
                "deleted_session_id": sid,
                "active_session_id": active_id,
                "created_replacement": created_replacement,
                "sessions": remaining_snapshot["sessions"],
            }

        if active_id == sid:
            replacement_meta = sorted(
                remaining,
                key=lambda x: str(x.get("updated_at") or ""),
                reverse=True,
            )[0]
            entry["active_session_id"] = str(replacement_meta.get("id"))

        self._save_index(index)
        snapshot = self.list_conversation_sessions(conversation_key)
        return {
            "deleted_session_id": sid,
            "active_session_id": snapshot["active_session_id"],
            "created_replacement": created_replacement,
            "sessions": snapshot["sessions"],
        }

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    legacy_path.rename(path)
                    logger.info(f"Migrated session {key} from legacy path")
                except Exception as e:
                    logger.warning(f"Failed migrating legacy session {key}: {e}")

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            updated_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        if data.get("created_at"):
                            created_at = datetime.fromisoformat(data["created_at"])
                        if data.get("updated_at"):
                            updated_at = datetime.fromisoformat(data["updated_at"])
                        last_consolidated = int(data.get("last_consolidated", 0) or 0)
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.key] = session

        split = self.split_composed_key(session.key)
        if split is not None:
            conversation_key, sid = split
            entry = self._ensure_conversation(conversation_key)
            meta = self._find_meta(entry, sid)
            if meta is None:
                meta = self._append_meta(
                    entry,
                    conversation_key,
                    sid,
                    title=session.metadata.get("title"),
                    auto_title=not bool((session.metadata.get("title") or "").strip()),
                )
            meta["updated_at"] = session.updated_at.isoformat()
            if title := session.metadata.get("title"):
                meta["title"] = self._normalize_title(str(title))
            self._save_index(self._load_index())

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions from disk.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = (
                                data.get("key")
                                if isinstance(data.get("key"), str)
                                else path.stem.replace("_", ":")
                            )
                            sessions.append(
                                {
                                    "key": key,
                                    "created_at": data.get("created_at"),
                                    "updated_at": data.get("updated_at"),
                                    "path": str(path),
                                }
                            )
            except Exception:
                continue

        return sorted(sessions, key=lambda x: str(x.get("updated_at", "")), reverse=True)
