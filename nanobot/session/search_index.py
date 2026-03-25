"""Rebuildable search index for file-backed session and memory artifacts."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import sqlite3
import tomllib
from pathlib import Path
from typing import Any

from nanobot.session.file_artifacts import SessionArtifactPaths
from nanobot.utils.helpers import ensure_dir, get_data_path


class SessionArtifactIndex:
    """Caches searchable metadata derived from local session and memory files."""

    _DB_NAME = "index.sqlite3"
    _SESSION_TRANSCRIPT_INDEX_MAX_CHARS = 20_000
    _MEMORY_ITEM_INDEX_MAX_CHARS = 8_000

    def __init__(self, workspace: Path):
        self.workspace = workspace
        data_root = get_data_path()
        self.root = ensure_dir(data_root / "cache")
        self.db_path = self.root / self._DB_NAME
        self.artifacts_root = data_root / "conversations"
        self.memory_root = data_root / "memories"
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS indexed_sessions (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    conversation_key TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    summary_text TEXT NOT NULL DEFAULT '',
                    working_set_text TEXT NOT NULL DEFAULT '',
                    transcript_text TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    app_server_thread_id TEXT NOT NULL DEFAULT '',
                    meta_path TEXT NOT NULL,
                    events_path TEXT NOT NULL,
                    summary_path TEXT NOT NULL,
                    working_set_path TEXT NOT NULL DEFAULT '',
                    transcript_path TEXT NOT NULL DEFAULT '',
                    source_hash TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS indexed_memory_items (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL UNIQUE,
                    scope TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    body_text TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    path TEXT NOT NULL,
                    source_hash TEXT NOT NULL DEFAULT ''
                );
                """
            )
            self._ensure_columns(
                conn,
                "indexed_sessions",
                {
                    "working_set_text": "TEXT NOT NULL DEFAULT ''",
                    "transcript_text": "TEXT NOT NULL DEFAULT ''",
                    "working_set_path": "TEXT NOT NULL DEFAULT ''",
                    "transcript_path": "TEXT NOT NULL DEFAULT ''",
                },
            )
            self._ensure_columns(
                conn,
                "indexed_memory_items",
                {
                    "scope": "TEXT NOT NULL DEFAULT ''",
                    "kind": "TEXT NOT NULL DEFAULT ''",
                    "title": "TEXT NOT NULL DEFAULT ''",
                    "body_text": "TEXT NOT NULL DEFAULT ''",
                    "updated_at": "TEXT NOT NULL DEFAULT ''",
                    "path": "TEXT NOT NULL DEFAULT ''",
                    "source_hash": "TEXT NOT NULL DEFAULT ''",
                },
            )
            self._recreate_session_fts(conn)
            self._recreate_memory_fts(conn)

    @staticmethod
    def _ensure_columns(
        conn: sqlite3.Connection,
        table_name: str,
        columns: dict[str, str],
    ) -> None:
        existing = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for name, ddl in columns.items():
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {ddl}")

    @staticmethod
    def _recreate_session_fts(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS indexed_sessions_ai;
            DROP TRIGGER IF EXISTS indexed_sessions_ad;
            DROP TRIGGER IF EXISTS indexed_sessions_au;
            DROP TABLE IF EXISTS indexed_sessions_fts;

            CREATE VIRTUAL TABLE indexed_sessions_fts USING fts5(
                title,
                summary_text,
                working_set_text,
                transcript_text,
                session_key,
                conversation_key,
                content='indexed_sessions',
                content_rowid='doc_id'
            );

            CREATE TRIGGER indexed_sessions_ai AFTER INSERT ON indexed_sessions BEGIN
                INSERT INTO indexed_sessions_fts(
                    rowid,
                    title,
                    summary_text,
                    working_set_text,
                    transcript_text,
                    session_key,
                    conversation_key
                )
                VALUES (
                    new.doc_id,
                    new.title,
                    new.summary_text,
                    new.working_set_text,
                    new.transcript_text,
                    new.session_key,
                    new.conversation_key
                );
            END;

            CREATE TRIGGER indexed_sessions_ad AFTER DELETE ON indexed_sessions BEGIN
                INSERT INTO indexed_sessions_fts(
                    indexed_sessions_fts,
                    rowid,
                    title,
                    summary_text,
                    working_set_text,
                    transcript_text,
                    session_key,
                    conversation_key
                )
                VALUES (
                    'delete',
                    old.doc_id,
                    old.title,
                    old.summary_text,
                    old.working_set_text,
                    old.transcript_text,
                    old.session_key,
                    old.conversation_key
                );
            END;

            CREATE TRIGGER indexed_sessions_au AFTER UPDATE ON indexed_sessions BEGIN
                INSERT INTO indexed_sessions_fts(
                    indexed_sessions_fts,
                    rowid,
                    title,
                    summary_text,
                    working_set_text,
                    transcript_text,
                    session_key,
                    conversation_key
                )
                VALUES (
                    'delete',
                    old.doc_id,
                    old.title,
                    old.summary_text,
                    old.working_set_text,
                    old.transcript_text,
                    old.session_key,
                    old.conversation_key
                );
                INSERT INTO indexed_sessions_fts(
                    rowid,
                    title,
                    summary_text,
                    working_set_text,
                    transcript_text,
                    session_key,
                    conversation_key
                )
                VALUES (
                    new.doc_id,
                    new.title,
                    new.summary_text,
                    new.working_set_text,
                    new.transcript_text,
                    new.session_key,
                    new.conversation_key
                );
            END;
            """
        )
        conn.execute("INSERT INTO indexed_sessions_fts(indexed_sessions_fts) VALUES ('rebuild')")

    @staticmethod
    def _recreate_memory_fts(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS indexed_memory_items_ai;
            DROP TRIGGER IF EXISTS indexed_memory_items_ad;
            DROP TRIGGER IF EXISTS indexed_memory_items_au;
            DROP TABLE IF EXISTS indexed_memory_items_fts;

            CREATE VIRTUAL TABLE indexed_memory_items_fts USING fts5(
                title,
                body_text,
                scope,
                kind,
                item_id,
                content='indexed_memory_items',
                content_rowid='doc_id'
            );

            CREATE TRIGGER indexed_memory_items_ai AFTER INSERT ON indexed_memory_items BEGIN
                INSERT INTO indexed_memory_items_fts(rowid, title, body_text, scope, kind, item_id)
                VALUES (new.doc_id, new.title, new.body_text, new.scope, new.kind, new.item_id);
            END;

            CREATE TRIGGER indexed_memory_items_ad AFTER DELETE ON indexed_memory_items BEGIN
                INSERT INTO indexed_memory_items_fts(
                    indexed_memory_items_fts,
                    rowid,
                    title,
                    body_text,
                    scope,
                    kind,
                    item_id
                )
                VALUES (
                    'delete',
                    old.doc_id,
                    old.title,
                    old.body_text,
                    old.scope,
                    old.kind,
                    old.item_id
                );
            END;

            CREATE TRIGGER indexed_memory_items_au AFTER UPDATE ON indexed_memory_items BEGIN
                INSERT INTO indexed_memory_items_fts(
                    indexed_memory_items_fts,
                    rowid,
                    title,
                    body_text,
                    scope,
                    kind,
                    item_id
                )
                VALUES (
                    'delete',
                    old.doc_id,
                    old.title,
                    old.body_text,
                    old.scope,
                    old.kind,
                    old.item_id
                );
                INSERT INTO indexed_memory_items_fts(rowid, title, body_text, scope, kind, item_id)
                VALUES (new.doc_id, new.title, new.body_text, new.scope, new.kind, new.item_id);
            END;
            """
        )
        conn.execute("INSERT INTO indexed_memory_items_fts(indexed_memory_items_fts) VALUES ('rebuild')")

    def upsert_session(self, paths: SessionArtifactPaths) -> None:
        """Index one session from its artifact files."""
        if not paths.meta_file.exists():
            return

        record = self._record_from_paths(paths)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO indexed_sessions (
                    session_id,
                    conversation_key,
                    session_key,
                    title,
                    summary_text,
                    working_set_text,
                    transcript_text,
                    updated_at,
                    app_server_thread_id,
                    meta_path,
                    events_path,
                    summary_path,
                    working_set_path,
                    transcript_path,
                    source_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    conversation_key = excluded.conversation_key,
                    session_key = excluded.session_key,
                    title = excluded.title,
                    summary_text = excluded.summary_text,
                    working_set_text = excluded.working_set_text,
                    transcript_text = excluded.transcript_text,
                    updated_at = excluded.updated_at,
                    app_server_thread_id = excluded.app_server_thread_id,
                    meta_path = excluded.meta_path,
                    events_path = excluded.events_path,
                    summary_path = excluded.summary_path,
                    working_set_path = excluded.working_set_path,
                    transcript_path = excluded.transcript_path,
                    source_hash = excluded.source_hash
                """,
                (
                    record["session_id"],
                    record["conversation_key"],
                    record["session_key"],
                    record["title"],
                    record["summary_text"],
                    record["working_set_text"],
                    record["transcript_text"],
                    record["updated_at"],
                    record["app_server_thread_id"],
                    record["meta_path"],
                    record["events_path"],
                    record["summary_path"],
                    record["working_set_path"],
                    record["transcript_path"],
                    record["source_hash"],
                ),
            )

    def upsert_memory_item(self, path: Path) -> None:
        """Index one itemized memory markdown file."""
        if not path.exists():
            return
        record = self._memory_record_from_path(path)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO indexed_memory_items (
                    item_id,
                    scope,
                    kind,
                    title,
                    body_text,
                    updated_at,
                    path,
                    source_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    scope = excluded.scope,
                    kind = excluded.kind,
                    title = excluded.title,
                    body_text = excluded.body_text,
                    updated_at = excluded.updated_at,
                    path = excluded.path,
                    source_hash = excluded.source_hash
                """,
                (
                    record["item_id"],
                    record["scope"],
                    record["kind"],
                    record["title"],
                    record["body_text"],
                    record["updated_at"],
                    record["path"],
                    record["source_hash"],
                ),
            )

    def rebuild(self) -> int:
        """Rebuild the session and memory indexes from current artifact files."""
        seen_session_ids: set[str] = set()
        allowed_session_ids_by_conversation: dict[Path, set[str] | None] = {}
        if self.artifacts_root.exists():
            for state_path in sorted(self.artifacts_root.glob("*/state.toml")):
                try:
                    state = tomllib.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    allowed_session_ids_by_conversation[state_path.parent] = None
                    continue
                sessions = state.get("sessions")
                if not isinstance(sessions, list):
                    allowed_session_ids_by_conversation[state_path.parent] = None
                    continue
                allowed_ids = {
                    str(item.get("id"))
                    for item in sessions
                    if isinstance(item, dict) and str(item.get("id") or "").strip()
                }
                allowed_session_ids_by_conversation[state_path.parent] = allowed_ids
        if self.artifacts_root.exists():
            for meta_path in sorted(self.artifacts_root.glob("*/sessions/*/meta.toml")):
                conversation_dir = meta_path.parent.parent.parent
                allowed_session_ids = allowed_session_ids_by_conversation.get(conversation_dir)
                session_id = meta_path.parent.name
                if allowed_session_ids is not None and session_id not in allowed_session_ids:
                    continue
                paths = SessionArtifactPaths(
                    conversation_dir=conversation_dir,
                    state_file=conversation_dir / "state.toml",
                    session_dir=meta_path.parent,
                    meta_file=meta_path,
                    events_file=meta_path.parent / "events.jsonl",
                    summary_file=meta_path.parent / "summary.md",
                    working_set_file=meta_path.parent / "working_set.md",
                    transcript_file=meta_path.parent / "transcript.md",
                )
                record = self._record_from_paths(paths)
                seen_session_ids.add(record["session_id"])
                self.upsert_session(paths)

        with self._connect() as conn:
            if seen_session_ids:
                placeholders = ", ".join("?" for _ in seen_session_ids)
                conn.execute(
                    f"DELETE FROM indexed_sessions WHERE session_id NOT IN ({placeholders})",
                    tuple(sorted(seen_session_ids)),
                )
            else:
                conn.execute("DELETE FROM indexed_sessions")

        self.rebuild_memory_index()
        return len(seen_session_ids)

    def rebuild_memory_index(self) -> int:
        """Rebuild the itemized memory index from `.nanobot/memory`."""
        seen_item_ids: set[str] = set()
        if self.memory_root.exists():
            for path in sorted(self.memory_root.rglob("*.md")):
                record = self._memory_record_from_path(path)
                seen_item_ids.add(record["item_id"])
                self.upsert_memory_item(path)

        with self._connect() as conn:
            if seen_item_ids:
                placeholders = ", ".join("?" for _ in seen_item_ids)
                conn.execute(
                    f"DELETE FROM indexed_memory_items WHERE item_id NOT IN ({placeholders})",
                    tuple(sorted(seen_item_ids)),
                )
            else:
                conn.execute("DELETE FROM indexed_memory_items")
        return len(seen_item_ids)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        conversation_key: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_empty("indexed_sessions"):
            self.rebuild()

        with self._connect() as conn:
            params: list[Any] = []
            if query.strip():
                match_query = self._fts_query(query)
                sql = """
                    SELECT s.*, bm25(indexed_sessions_fts) AS score
                    FROM indexed_sessions_fts
                    JOIN indexed_sessions s ON s.doc_id = indexed_sessions_fts.rowid
                    WHERE indexed_sessions_fts MATCH ?
                """
                params.append(match_query)
                if conversation_key:
                    sql += " AND s.conversation_key = ?"
                    params.append(conversation_key)
                sql += " ORDER BY score ASC, s.updated_at DESC LIMIT ?"
                params.append(max(1, limit))
                rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                sql = "SELECT s.*, 1.0 AS score FROM indexed_sessions s"
                if conversation_key:
                    sql += " WHERE s.conversation_key = ?"
                    params.append(conversation_key)
                sql += " ORDER BY s.updated_at DESC LIMIT ?"
                params.append(max(1, limit))
                rows = conn.execute(sql, tuple(params)).fetchall()

        lowered = query.strip().lower()
        return [
            {
                "session_id": row["session_id"],
                "id": row["session_id"],
                "key": row["session_key"],
                "conversation_key": row["conversation_key"],
                "title": row["title"],
                "summary": row["summary_text"],
                "updated_at": row["updated_at"],
                "score": self._normalized_score(row["score"]),
                "snippet": self._best_snippet(
                    lowered,
                    [
                        row["working_set_text"],
                        row["summary_text"],
                        row["transcript_text"],
                        row["title"],
                        row["session_key"],
                    ],
                ),
            }
            for row in rows
        ]

    def search_memory(
        self,
        query: str,
        *,
        limit: int = 10,
        scope: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_empty("indexed_memory_items"):
            self.rebuild_memory_index()

        with self._connect() as conn:
            params: list[Any] = []
            rows: list[sqlite3.Row]
            if query.strip():
                match_query = self._fts_query(query)
                sql = """
                    SELECT m.*, bm25(indexed_memory_items_fts) AS score
                    FROM indexed_memory_items_fts
                    JOIN indexed_memory_items m ON m.doc_id = indexed_memory_items_fts.rowid
                    WHERE indexed_memory_items_fts MATCH ?
                """
                params.append(match_query)
                if scope:
                    sql += " AND m.scope = ?"
                    params.append(scope)
                if kind:
                    sql += " AND m.kind = ?"
                    params.append(kind)
                sql += " ORDER BY score ASC, m.updated_at DESC LIMIT ?"
                params.append(max(1, limit))
                rows = conn.execute(sql, tuple(params)).fetchall()
                if not rows:
                    rows = self._search_memory_fallback(
                        conn,
                        query=query,
                        limit=limit,
                        scope=scope,
                        kind=kind,
                    )
            else:
                sql = "SELECT m.*, 1.0 AS score FROM indexed_memory_items m WHERE 1 = 1"
                if scope:
                    sql += " AND m.scope = ?"
                    params.append(scope)
                if kind:
                    sql += " AND m.kind = ?"
                    params.append(kind)
                sql += " ORDER BY m.updated_at DESC LIMIT ?"
                params.append(max(1, limit))
                rows = conn.execute(sql, tuple(params)).fetchall()

        lowered = query.strip().lower()
        return [
            {
                "item_id": row["item_id"],
                "scope": row["scope"],
                "kind": row["kind"],
                "title": row["title"],
                "updated_at": row["updated_at"],
                "path": row["path"],
                "score": self._normalized_score(row["score"]),
                "snippet": self._best_snippet(lowered, [row["title"], row["body_text"]]),
            }
            for row in rows
        ]

    def read_memory_item(self, item_id: str) -> dict[str, Any] | None:
        row = self._get_memory_row(item_id)
        if row is None:
            self.rebuild_memory_index()
            row = self._get_memory_row(item_id)
        if row is None:
            return None

        path = Path(row["path"])
        if not path.exists():
            return None
        return {
            "item": {
                "item_id": row["item_id"],
                "scope": row["scope"],
                "kind": row["kind"],
                "title": row["title"],
                "updated_at": row["updated_at"],
                "path": row["path"],
            },
            "content": path.read_text(encoding="utf-8"),
        }

    def read_session(self, session_id: str, *, mode: str = "summary", limit: int = 50) -> dict[str, Any] | None:
        row = self._get_index_row(session_id)
        if row is None:
            self.rebuild()
            row = self._get_index_row(session_id)
        if row is None:
            return None

        self._refresh_if_stale(row)
        row = self._get_index_row(session_id)
        if row is None:
            return None

        meta_path = Path(row["meta_path"])
        events_path = Path(row["events_path"])
        summary_path = Path(row["summary_path"])
        working_set_path = Path(row["working_set_path"])
        transcript_path = Path(row["transcript_path"])

        meta = self._load_meta(meta_path)
        title = str(meta.get("title") or row["title"] or "").strip()
        summary = self._read_summary(summary_path) or str(meta.get("summary") or "").strip()
        updated_at = str(meta.get("updated_at") or row["updated_at"] or "")

        content: Any
        if mode == "messages":
            content = self._read_messages(events_path, limit=max(1, limit))
        elif mode == "snippet":
            messages = self._read_messages(events_path, limit=max(1, limit))
            lines = []
            for msg in messages:
                role = str(msg.get("role") or "").upper()
                text = str(msg.get("content") or "").strip()
                if text:
                    lines.append(f"{role}: {text}")
            content = "\n".join(lines) or summary or self._read_summary(summary_path)
        elif mode == "working_set":
            content = self._read_file_raw(working_set_path)
        elif mode == "transcript":
            content = self._read_file_raw(transcript_path)
        else:
            content = summary or self._read_summary(summary_path)

        return {
            "session": {
                "id": session_id,
                "key": row["session_key"],
                "conversation_key": row["conversation_key"],
                "title": title,
                "summary": summary,
                "updated_at": updated_at,
            },
            "mode": mode,
            "content": content,
        }

    def _refresh_if_stale(self, row: sqlite3.Row) -> None:
        meta_path = Path(row["meta_path"])
        events_path = Path(row["events_path"])
        summary_path = Path(row["summary_path"])
        working_set_path = Path(row["working_set_path"])
        transcript_path = Path(row["transcript_path"])
        current_hash = self._source_hash(
            meta_path,
            events_path,
            summary_path,
            working_set_path,
            transcript_path,
        )
        if current_hash == row["source_hash"]:
            return
        if not meta_path.exists():
            return
        paths = SessionArtifactPaths(
            conversation_dir=meta_path.parent.parent.parent,
            state_file=meta_path.parent.parent.parent / "state.toml",
            session_dir=meta_path.parent,
            meta_file=meta_path,
            events_file=events_path,
            summary_file=summary_path,
            working_set_file=working_set_path,
            transcript_file=transcript_path,
        )
        self.upsert_session(paths)

    def _record_from_paths(self, paths: SessionArtifactPaths) -> dict[str, str]:
        meta = self._load_meta(paths.meta_file)
        summary_text = self._read_summary(paths.summary_file) or str(meta.get("summary") or "").strip()
        working_set_text = self._read_text(paths.working_set_file, max_chars=6_000)
        transcript_text = self._read_text(
            paths.transcript_file,
            max_chars=self._SESSION_TRANSCRIPT_INDEX_MAX_CHARS,
        )
        metadata = meta.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "session_id": str(meta.get("id") or paths.session_dir.name),
            "conversation_key": str(meta.get("conversation_key") or ""),
            "session_key": str(meta.get("key") or ""),
            "title": str(meta.get("title") or "").strip(),
            "summary_text": summary_text.strip(),
            "working_set_text": working_set_text.strip(),
            "transcript_text": transcript_text.strip(),
            "updated_at": str(meta.get("updated_at") or ""),
            "app_server_thread_id": str(metadata.get("app_server_thread_id") or "").strip(),
            "meta_path": str(paths.meta_file),
            "events_path": str(paths.events_file),
            "summary_path": str(paths.summary_file),
            "working_set_path": str(paths.working_set_file),
            "transcript_path": str(paths.transcript_file),
            "source_hash": self._source_hash(
                paths.meta_file,
                paths.events_file,
                paths.summary_file,
                paths.working_set_file,
                paths.transcript_file,
            ),
        }

    def _memory_record_from_path(self, path: Path) -> dict[str, str]:
        rel = path.relative_to(self.memory_root).as_posix() if path.is_relative_to(self.memory_root) else path.name
        scope, kind = self._memory_scope_and_kind(rel)
        text = self._read_text(path, max_chars=self._MEMORY_ITEM_INDEX_MAX_CHARS)
        title = self._extract_title(path, text)
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        return {
            "item_id": rel,
            "scope": scope,
            "kind": kind,
            "title": title,
            "body_text": text.strip(),
            "updated_at": updated_at,
            "path": str(path),
            "source_hash": self._source_hash(path),
        }

    @staticmethod
    def _memory_scope_and_kind(rel_path: str) -> tuple[str, str]:
        parts = Path(rel_path).parts
        if not parts:
            return "", ""
        if parts[0] == "global":
            kind = parts[1] if len(parts) >= 2 else "memory"
            return "global", kind
        if parts[0] == "workspaces":
            workspace_id = parts[1] if len(parts) >= 2 else "workspace"
            kind = parts[2] if len(parts) >= 3 else "memory"
            return f"workspace:{workspace_id}", kind
        return parts[0], parts[1] if len(parts) >= 2 else "memory"

    def _get_index_row(self, session_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM indexed_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

    def _get_memory_row(self, item_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM indexed_memory_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()

    def _is_empty(self, table_name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
        return int(row["count"] or 0) == 0

    @staticmethod
    def _load_meta(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _read_summary(path: Path) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        marker = "## Summary"
        if marker not in text:
            return text
        _head, _sep, tail = text.partition(marker)
        return tail.strip().lstrip("#").strip()

    @staticmethod
    def _read_text(path: Path, *, max_chars: int | None = None) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8").strip()
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text

    @staticmethod
    def _read_messages(path: Path, limit: int) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        messages: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                messages.append(payload)
        return messages[-max(1, limit):]

    @staticmethod
    def _extract_title(path: Path, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.stem
        return path.stem.replace("-", " ").replace("_", " ").strip() or path.stem

    @staticmethod
    def _read_file_raw(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _best_snippet(query: str, candidates: list[Any]) -> str:
        normalized = query.strip().lower()
        for candidate in candidates:
            text = " ".join(str(candidate or "").split())
            if not text:
                continue
            if normalized and normalized in text.lower():
                idx = text.lower().find(normalized)
                start = max(0, idx - 80)
                end = min(len(text), idx + max(80, len(normalized) + 80))
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet += "..."
                return snippet
            if not normalized:
                return text[:200]
        return ""

    @staticmethod
    def _search_memory_fallback(
        conn: sqlite3.Connection,
        *,
        query: str,
        limit: int,
        scope: str | None,
        kind: str | None,
    ) -> list[sqlite3.Row]:
        like = f"%{query.strip()}%"
        params: list[Any] = [like, like, like, like]
        sql = """
            SELECT m.*, 1.0 AS score
            FROM indexed_memory_items m
            WHERE (
                m.title LIKE ?
                OR m.body_text LIKE ?
                OR m.item_id LIKE ?
                OR m.path LIKE ?
            )
        """
        if scope:
            sql += " AND m.scope = ?"
            params.append(scope)
        if kind:
            sql += " AND m.kind = ?"
            params.append(kind)
        sql += " ORDER BY m.updated_at DESC LIMIT ?"
        params.append(max(1, limit))
        return conn.execute(sql, tuple(params)).fetchall()

    @staticmethod
    def _source_hash(*paths: Path) -> str:
        h = hashlib.sha256()
        for path in paths:
            h.update(str(path).encode("utf-8"))
            if path.exists():
                h.update(path.read_bytes())
        return h.hexdigest()

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = []
        for token in query.split():
            token = token.strip().replace('"', "")
            if token:
                terms.append(f'"{token}"')
        return " AND ".join(terms) if terms else '""'

    @staticmethod
    def _normalized_score(raw_score: Any) -> float:
        try:
            score = float(raw_score)
        except Exception:
            return 1.0
        if score < 0:
            return 1.0 / (1.0 + abs(score))
        return 1.0 / (1.0 + score)
