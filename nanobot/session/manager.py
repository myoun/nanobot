"""Session management for conversation history."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.session.file_artifacts import SessionArtifactStore
from nanobot.session.search_index import SessionArtifactIndex
from nanobot.utils.helpers import ensure_dir, get_data_path, safe_filename


@dataclass
class Session:
    """
    A conversation thread/session.

    Messages stay append-only in memory. Persistence is handled by SessionManager.
    """

    key: str
    id: str | None = None
    conversation_key: str | None = None
    conversation_id: str | None = None
    title: str = ""
    summary: str = ""
    kind: str = "chat"
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0

    @staticmethod
    def _normalize_call_id(raw: Any) -> str:
        """Normalize call IDs like `call_id|item_id` to `call_id`."""
        if not isinstance(raw, str) or not raw:
            return ""
        return raw.split("|", 1)[0]

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
        """Get recent unconsolidated messages, aligned to a user turn and valid tool chains."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        for i, m in enumerate(sliced):
            if m.get("role") == "user":
                sliced = sliced[i:]
                break

        out: list[dict[str, Any]] = []
        seen_call_ids: set[str] = set()

        for m in sliced:
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            role = m.get("role")
            if role == "assistant":
                tool_calls = m.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        call_id = self._normalize_call_id(tc.get("id"))
                        if call_id:
                            seen_call_ids.add(call_id)

            if role == "tool":
                call_id = self._normalize_call_id(m.get("tool_call_id"))
                if not call_id or call_id not in seen_call_ids:
                    continue

            entry: dict[str, Any] = {"role": m["role"], "content": content}
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
    """Manages conversations and sessions/threads using SQLite."""

    _DB_NAME = "sessions.sqlite3"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        try:
            workspace_resolved = workspace.resolve()
            workspace_name = workspace_resolved.name
            workspace_fingerprint = hashlib.sha1(
                str(workspace_resolved).encode("utf-8")
            ).hexdigest()[:10]
        except Exception:
            workspace_name = workspace.name if isinstance(workspace.name, str) else "workspace"
            workspace_fingerprint = "workspace"
        workspace_id = safe_filename(f"{workspace_name or 'workspace'}-{workspace_fingerprint}")
        self.data_root = ensure_dir(get_data_path() / "workspaces" / workspace_id)
        self.db_path = self.data_root / self._DB_NAME
        self.artifacts = SessionArtifactStore(workspace)
        self.index = SessionArtifactIndex(workspace)
        self._cache: dict[str, Session] = {}
        self._cache_by_id: dict[str, str] = {}
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    conversation_key TEXT NOT NULL UNIQUE,
                    channel TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    active_session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    session_key TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT 'chat',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    last_consolidated INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, position),
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_conversation_id ON sessions(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);
                """
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _short_id() -> str:
        return uuid.uuid4().hex[:8]

    @staticmethod
    def _conversation_key_from_session_key(key: str) -> str | None:
        if "#" in key:
            head, _sep, _tail = key.partition("#")
            return head or None
        if ":" in key:
            return key
        return None

    @staticmethod
    def _parse_conversation_key(conversation_key: str) -> tuple[str, str]:
        if ":" not in conversation_key:
            return conversation_key, ""
        return conversation_key.split(":", 1)

    def _session_info_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        conversation_key = row["conversation_key"] if "conversation_key" in row.keys() else None
        return {
            "id": row["id"],
            "key": row["session_key"],
            "conversation_id": row["conversation_id"],
            "conversation_key": conversation_key or self._conversation_key_from_session_key(row["session_key"]),
            "title": row["title"],
            "summary": row["summary"],
            "kind": row["kind"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _conversation_info_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "key": row["conversation_key"],
            "conversation_key": row["conversation_key"],
            "channel": row["channel"],
            "chat_id": row["chat_id"],
            "active_session_id": row["active_session_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _load_messages(self, conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT payload_json FROM messages WHERE session_id = ? ORDER BY position ASC",
            (session_id,),
        ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except Exception:
                payload = {"role": "assistant", "content": row["payload_json"]}
            if isinstance(payload, dict):
                messages.append(payload)
        return messages

    def _session_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Session:
        metadata_raw = row["metadata_json"] or "{}"
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
        session = Session(
            key=row["session_key"],
            id=row["id"],
            conversation_key=row["conversation_key"] if "conversation_key" in row.keys() else self._conversation_key_from_session_key(row["session_key"]),
            conversation_id=row["conversation_id"],
            title=row["title"],
            summary=row["summary"],
            kind=row["kind"],
            messages=self._load_messages(conn, row["id"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=metadata if isinstance(metadata, dict) else {},
            last_consolidated=int(row["last_consolidated"] or 0),
        )
        self._cache[session.key] = session
        if session.id:
            self._cache_by_id[session.id] = session.key
        return session

    def _get_session_row_by_key(self, conn: sqlite3.Connection, key: str) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT s.*, c.conversation_key
            FROM sessions s
            LEFT JOIN conversations c ON c.id = s.conversation_id
            WHERE s.session_key = ?
            """,
            (key,),
        ).fetchone()

    def _get_session_row_by_id(self, conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT s.*, c.conversation_key
            FROM sessions s
            LEFT JOIN conversations c ON c.id = s.conversation_id
            WHERE s.id = ?
            """,
            (session_id,),
        ).fetchone()

    def _create_session_row(
        self,
        conn: sqlite3.Connection,
        *,
        session_key: str,
        conversation_id: str | None,
        title: str = "",
        summary: str = "",
        kind: str = "chat",
        metadata: dict[str, Any] | None = None,
        last_consolidated: int = 0,
        created_at: str | None = None,
        updated_at: str | None = None,
        session_id: str | None = None,
    ) -> str:
        now = self._now_iso()
        sid = session_id or self._short_id()
        conn.execute(
            """
            INSERT INTO sessions (
                id, conversation_id, session_key, title, summary, kind,
                metadata_json, last_consolidated, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                conversation_id,
                session_key,
                title,
                summary,
                kind,
                json.dumps(metadata or {}, ensure_ascii=False),
                int(last_consolidated),
                created_at or now,
                updated_at or now,
            ),
        )
        return sid

    def _ensure_conversation(self, conversation_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_key = ?",
                (conversation_key,),
            ).fetchone()
            if not row:
                channel, chat_id = self._parse_conversation_key(conversation_key)
                conversation_id = self._short_id()
                now = self._now_iso()
                conn.execute(
                    """
                    INSERT INTO conversations (
                        id, conversation_key, channel, chat_id, active_session_id, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (conversation_id, conversation_key, channel, chat_id, now, now),
                )
                existing = self._get_session_row_by_key(conn, conversation_key)
                if existing:
                    session_id = existing["id"]
                    conn.execute(
                        "UPDATE sessions SET conversation_id = ? WHERE id = ?",
                        (conversation_id, session_id),
                    )
                else:
                    session_id = self._create_session_row(
                        conn,
                        session_key=conversation_key,
                        conversation_id=conversation_id,
                        title="",
                    )
                conn.execute(
                    "UPDATE conversations SET active_session_id = ?, updated_at = ? WHERE id = ?",
                    (session_id, now, conversation_id),
                )
                row = conn.execute(
                    "SELECT * FROM conversations WHERE conversation_key = ?",
                    (conversation_key,),
                ).fetchone()
            conversation = self._conversation_info_from_row(row)

            active_id = conversation.get("active_session_id")
            active_row = self._get_session_row_by_id(conn, active_id) if active_id else None
            if active_row is None:
                base_row = self._get_session_row_by_key(conn, conversation_key)
                if base_row is None:
                    session_id = self._create_session_row(
                        conn,
                        session_key=conversation_key,
                        conversation_id=conversation["id"],
                    )
                    base_row = self._get_session_row_by_id(conn, session_id)
                conn.execute(
                    "UPDATE conversations SET active_session_id = ?, updated_at = ? WHERE id = ?",
                    (base_row["id"], self._now_iso(), conversation["id"]),
                )
                conversation["active_session_id"] = base_row["id"]
                active_row = base_row

            return conversation, self._session_info_from_row(active_row)

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]

        with self._connect() as conn:
            row = self._get_session_row_by_key(conn, key)
            if row:
                return self._session_from_row(conn, row)

        if "#" not in key and ":" in key:
            self._ensure_conversation(key)
            with self._connect() as conn:
                row = self._get_session_row_by_key(conn, key)
                if row:
                    return self._session_from_row(conn, row)

        session = Session(key=key, conversation_key=self._conversation_key_from_session_key(key))
        self.save(session)
        return self._cache[key]

    def get_by_id(self, session_id: str) -> Session | None:
        if key := self._cache_by_id.get(session_id):
            return self._cache.get(key)

        with self._connect() as conn:
            row = self._get_session_row_by_id(conn, session_id)
            if row:
                return self._session_from_row(conn, row)
        return None

    def get_or_create_for_conversation(self, conversation_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        snapshot = self._ensure_conversation(conversation_key)
        self._sync_conversation_artifacts(conversation_key)
        return snapshot

    def get_active_session(self, conversation_key: str) -> Session:
        _conversation, active = self._ensure_conversation(conversation_key)
        session = self.get_or_create(str(active["key"]))
        self._sync_session_artifacts(session)
        self._sync_conversation_artifacts(conversation_key)
        return session

    def create_session(
        self,
        conversation_key: str,
        title: str | None = None,
        *,
        kind: str = "chat",
        switch_to: bool = True,
    ) -> dict[str, Any]:
        conversation, _active = self._ensure_conversation(conversation_key)
        with self._connect() as conn:
            session_id = self._short_id()
            session_key = f"{conversation_key}#{session_id}"
            now = self._now_iso()
            self._create_session_row(
                conn,
                session_id=session_id,
                session_key=session_key,
                conversation_id=conversation["id"],
                title=title or "",
                kind=kind,
                created_at=now,
                updated_at=now,
            )
            if switch_to:
                conn.execute(
                    "UPDATE conversations SET active_session_id = ?, updated_at = ? WHERE id = ?",
                    (session_id, now, conversation["id"]),
                )
            row = self._get_session_row_by_id(conn, session_id)
            info = self._session_info_from_row(row)
        self.invalidate(info["key"])
        if session := self.get_by_id(str(info["id"])):
            self._sync_session_artifacts(session)
        self._sync_conversation_artifacts(conversation_key)
        return info

    def switch_session(self, conversation_key: str, session_id: str) -> dict[str, Any]:
        conversation, _active = self._ensure_conversation(conversation_key)
        with self._connect() as conn:
            row = self._get_session_row_by_id(conn, session_id)
            if not row or row["conversation_id"] != conversation["id"]:
                raise ValueError(f"Session {session_id} does not belong to {conversation_key}")
            conn.execute(
                "UPDATE conversations SET active_session_id = ?, updated_at = ? WHERE id = ?",
                (session_id, self._now_iso(), conversation["id"]),
            )
            info = self._session_info_from_row(row)
        self._sync_conversation_artifacts(conversation_key)
        return info

    def list_conversation_sessions(self, conversation_key: str) -> dict[str, Any]:
        conversation, _active = self._ensure_conversation(conversation_key)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, c.conversation_key
                FROM sessions s
                LEFT JOIN conversations c ON c.id = s.conversation_id
                WHERE s.conversation_id = ?
                ORDER BY s.updated_at DESC, s.created_at DESC
                """,
                (conversation["id"],),
            ).fetchall()
        return {
            "conversation": conversation,
            "active_session_id": conversation["active_session_id"],
            "sessions": [self._session_info_from_row(row) for row in rows],
        }

    def save(self, session: Session) -> None:
        self.artifacts.refresh_summary_checkpoint(session)
        conversation_key = session.conversation_key or self._conversation_key_from_session_key(session.key)
        now = self._now_iso()
        with self._connect() as conn:
            row = self._get_session_row_by_key(conn, session.key)
            if row:
                session.id = row["id"]
                session.conversation_id = row["conversation_id"]
            elif session.id:
                row = self._get_session_row_by_id(conn, session.id)
                if row:
                    session.key = row["session_key"]
                    session.conversation_id = row["conversation_id"]

            if row is None:
                conversation_id = None
                if "#" not in session.key and conversation_key == session.key and ":" in session.key:
                    conversation, active = self._ensure_conversation(session.key)
                    conversation_id = conversation["id"]
                    existing = self.get_or_create(str(active["key"]))
                    session.id = existing.id
                    session.conversation_id = existing.conversation_id
                    row = None
                elif conversation_key and "#" in session.key:
                    conversation, _active = self._ensure_conversation(conversation_key)
                    conversation_id = conversation["id"]

                if session.id:
                    conn.execute(
                        """
                        UPDATE sessions
                        SET conversation_id = COALESCE(?, conversation_id),
                            title = ?, summary = ?, kind = ?, metadata_json = ?,
                            last_consolidated = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            session.conversation_id or conversation_id,
                            session.title,
                            session.summary,
                            session.kind,
                            json.dumps(session.metadata, ensure_ascii=False),
                            int(session.last_consolidated),
                            session.updated_at.isoformat(),
                            session.id,
                        ),
                    )
                else:
                    session.id = self._create_session_row(
                        conn,
                        session_key=session.key,
                        conversation_id=conversation_id,
                        title=session.title,
                        summary=session.summary,
                        kind=session.kind,
                        metadata=session.metadata,
                        last_consolidated=session.last_consolidated,
                        created_at=session.created_at.isoformat(),
                        updated_at=session.updated_at.isoformat(),
                    )
                    session.conversation_id = conversation_id
            else:
                session.id = row["id"]
                session.conversation_id = row["conversation_id"]
                conn.execute(
                    """
                    UPDATE sessions
                    SET title = ?, summary = ?, kind = ?, metadata_json = ?,
                        last_consolidated = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        session.title,
                        session.summary,
                        session.kind,
                        json.dumps(session.metadata, ensure_ascii=False),
                        int(session.last_consolidated),
                        session.updated_at.isoformat(),
                        session.id,
                    ),
                )

            if conversation_key and session.conversation_id:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, session.conversation_id),
                )

            conn.execute("DELETE FROM messages WHERE session_id = ?", (session.id,))
            for idx, msg in enumerate(session.messages):
                conn.execute(
                    "INSERT INTO messages (session_id, position, payload_json) VALUES (?, ?, ?)",
                    (session.id, idx, json.dumps(msg, ensure_ascii=False)),
                )

        self._cache[session.key] = session
        if session.id:
            self._cache_by_id[session.id] = session.key
        self._sync_session_artifacts(session)
        if session.conversation_key:
            self._sync_conversation_artifacts(session.conversation_key)

    def invalidate(self, key: str) -> None:
        session = self._cache.pop(key, None)
        if session and session.id:
            self._cache_by_id.pop(session.id, None)

    def _sync_session_artifacts(self, session: Session) -> None:
        """Mirror a session into file artifacts."""
        try:
            paths = self.artifacts.write_session(session)
            if paths is not None:
                self.index.upsert_session(paths)
        except Exception as e:
            logger.warning("Failed to write session artifacts for {}: {}", session.key, e)

    def _sync_conversation_artifacts(self, conversation_key: str) -> None:
        """Mirror conversation state into file artifacts."""
        try:
            snapshot = self.list_conversation_sessions(conversation_key)
            self.artifacts.write_conversation_state(snapshot)
        except Exception as e:
            logger.warning("Failed to write conversation artifacts for {}: {}", conversation_key, e)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "key": row["conversation_key"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "active_session_id": row["active_session_id"],
            }
            for row in rows
        ]

    def rebuild_search_index(self) -> int:
        """Rebuild the file-derived search cache from disk."""
        snapshots: list[dict[str, Any]] = []
        for conversation in self.list_sessions():
            conversation_key = str(conversation.get("key") or "").strip()
            if not conversation_key:
                continue
            snapshots.extend(self.list_conversation_sessions(conversation_key).get("sessions", []))
        seen_session_ids: set[str] = set()
        for item in snapshots:
            session_id = str(item.get("id") or "").strip()
            conversation_key = str(item.get("conversation_key") or item.get("key") or "").strip()
            if not session_id or not conversation_key:
                continue
            seen_session_ids.add(session_id)
            self.index.upsert_session(self.artifacts.paths_for(conversation_key, session_id))

        with self.index._connect() as conn:
            if seen_session_ids:
                placeholders = ", ".join("?" for _ in seen_session_ids)
                conn.execute(
                    f"DELETE FROM indexed_sessions WHERE session_id NOT IN ({placeholders})",
                    tuple(sorted(seen_session_ids)),
                )
            else:
                conn.execute("DELETE FROM indexed_sessions")

        self.index.rebuild_memory_index()
        return len(seen_session_ids)

    def search_sessions(
        self,
        query: str,
        *,
        limit: int = 10,
        conversation_key: str | None = None,
    ) -> list[dict[str, Any]]:
        valid_session_ids: set[str] = set()
        conversations = [conversation_key] if conversation_key else [
            str(item.get("key") or "").strip()
            for item in self.list_sessions()
            if str(item.get("key") or "").strip()
        ]
        for key in conversations:
            valid_session_ids.update(
                str(item.get("id") or "").strip()
                for item in self.list_conversation_sessions(key).get("sessions", [])
                if str(item.get("id") or "").strip()
            )
        try:
            hits = self.index.search(
                query=query,
                limit=limit,
                conversation_key=conversation_key,
            )
            hits = [hit for hit in hits if str(hit.get("session_id") or "") in valid_session_ids]
            if hits or not query.strip():
                return hits
        except Exception as e:
            logger.warning("Artifact-backed session search failed for {}: {}", query, e)

        with self._connect() as conn:
            params: list[Any] = []
            where: list[str] = []
            if conversation_key:
                where.append("c.conversation_key = ?")
                params.append(conversation_key)
            if query.strip():
                like = f"%{query.strip()}%"
                where.append(
                    "(s.title LIKE ? OR s.summary LIKE ? OR s.session_key LIKE ? OR c.conversation_key LIKE ?)"
                )
                params.extend([like, like, like, like])
            sql = """
                SELECT s.*, c.conversation_key
                FROM sessions s
                LEFT JOIN conversations c ON c.id = s.conversation_id
            """
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY s.updated_at DESC LIMIT ?"
            params.append(max(1, limit))
            rows = conn.execute(sql, tuple(params)).fetchall()
        hits: list[dict[str, Any]] = []
        lowered = query.strip().lower()
        for row in rows:
            snippet = row["summary"] or row["title"] or row["session_key"]
            score = 1.0
            if lowered:
                haystacks = [
                    str(row["title"] or "").lower(),
                    str(row["summary"] or "").lower(),
                    str(row["session_key"] or "").lower(),
                    str(row["conversation_key"] or "").lower(),
                ]
                score = 2.0 if any(lowered in h for h in haystacks) else 1.0
            hits.append(
                {
                    "session_id": row["id"],
                    "id": row["id"],
                    "key": row["session_key"],
                    "conversation_key": row["conversation_key"] or self._conversation_key_from_session_key(row["session_key"]),
                    "title": row["title"],
                    "summary": row["summary"],
                    "updated_at": row["updated_at"],
                    "score": score,
                    "snippet": snippet,
                }
            )
        return hits

    def read_session(
        self,
        session_id: str,
        *,
        mode: str = "summary",
        limit: int = 50,
    ) -> dict[str, Any]:
        try:
            if artifact_result := self.index.read_session(session_id, mode=mode, limit=limit):
                return artifact_result
        except Exception as e:
            logger.warning("Artifact-backed session read failed for {}: {}", session_id, e)

        with self._connect() as conn:
            row = self._get_session_row_by_id(conn, session_id)
            if not row:
                raise ValueError(f"Unknown session {session_id}")
            session = self._session_from_row(conn, row)

        content: Any
        if mode == "messages":
            content = session.messages[-max(1, limit):]
        elif mode == "snippet":
            lines: list[str] = []
            for msg in session.messages[-max(1, limit):]:
                role = str(msg.get("role") or "").upper()
                text = str(msg.get("content") or "").strip()
                if not text:
                    continue
                lines.append(f"{role}: {text}")
            content = "\n".join(lines[-max(1, limit):]) or session.summary or session.title
        elif mode == "working_set":
            paths = self.artifacts.paths_for(
                session.conversation_key or self._conversation_key_from_session_key(session.key) or session.key,
                str(session.id or safe_filename(session.key)),
            )
            content = paths.working_set_file.read_text(encoding="utf-8") if paths.working_set_file.exists() else ""
        elif mode == "transcript":
            paths = self.artifacts.paths_for(
                session.conversation_key or self._conversation_key_from_session_key(session.key) or session.key,
                str(session.id or safe_filename(session.key)),
            )
            content = paths.transcript_file.read_text(encoding="utf-8") if paths.transcript_file.exists() else ""
        else:
            content = session.summary or session.title
            if not content:
                for msg in reversed(session.messages):
                    text = str(msg.get("content") or "").strip()
                    if text:
                        content = text
                        break
        return {
            "session": {
                "id": session.id,
                "key": session.key,
                "conversation_key": session.conversation_key,
                "title": session.title,
                "summary": session.summary,
                "updated_at": session.updated_at.isoformat(),
            },
            "mode": mode,
            "content": content,
        }
