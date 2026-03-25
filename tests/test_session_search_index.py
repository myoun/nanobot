from __future__ import annotations

import json
from pathlib import Path

from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import get_data_path


def _conversation_dir(root: Path, conversation_key: str) -> Path:
    return get_data_path() / "conversations" / conversation_key.replace(":", "__")


def test_search_sessions_uses_file_backed_index(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-index")
    session.title = "Rollback planning"
    session.summary = "Discussed the backfill, rollback, and audit plan."
    session.add_message("user", "Find the rollback notes.")
    session.add_message("assistant", "Captured the rollback plan.")
    manager.save(session)

    hits = manager.search_sessions("rollback", limit=5)

    assert session.id is not None
    assert hits
    assert hits[0]["session_id"] == session.id
    assert "rollback" in hits[0]["summary"].lower()
    assert (get_data_path() / "cache" / "index.sqlite3").exists()


def test_rebuild_search_index_picks_up_manual_summary_edits(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-rebuild")
    session.title = "DB migration"
    session.summary = "Initial migration notes."
    manager.save(session)
    assert session.id is not None

    session_dir = _conversation_dir(tmp_path, "cli:test-rebuild") / "sessions" / session.id
    summary_path = session_dir / "summary.md"
    summary_path.write_text(
        "# DB migration\n\n## Summary\n\nManual note about sentinel backfill recovery.\n",
        encoding="utf-8",
    )

    manager.rebuild_search_index()
    hits = manager.search_sessions("sentinel", limit=5)
    read_result = manager.read_session(session.id, mode="summary")

    assert hits
    assert hits[0]["session_id"] == session.id
    assert "sentinel backfill recovery" in hits[0]["summary"].lower()
    assert "Manual note about sentinel backfill recovery." in str(read_result["content"])


def test_read_session_messages_reads_file_artifact_contents(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-read-artifacts")
    session.title = "Artifact read"
    session.summary = "Original summary."
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    manager.save(session)
    assert session.id is not None

    session_dir = _conversation_dir(tmp_path, "cli:test-read-artifacts") / "sessions" / session.id
    events_path = session_dir / "events.jsonl"
    altered_events = [
        {"index": 0, "role": "user", "content": "manual user note"},
        {"index": 1, "role": "assistant", "content": "manual assistant note"},
    ]
    events_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in altered_events) + "\n",
        encoding="utf-8",
    )

    result = manager.read_session(session.id, mode="messages", limit=2)

    assert result["content"] == altered_events


def test_search_sessions_indexes_working_set_and_transcript(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-rich-index")
    session.title = "Rich search"
    session.summary = "Short summary."
    session.add_message("user", "initial request")
    manager.save(session)
    assert session.id is not None

    session_dir = _conversation_dir(tmp_path, "cli:test-rich-index") / "sessions" / session.id
    (session_dir / "working_set.md").write_text(
        "# Working Set\n\nCurrent blocker is the sentinel rehydration bug.\n",
        encoding="utf-8",
    )
    (session_dir / "transcript.md").write_text(
        "# Transcript\n\nHistoric note: rollback cursor drift was fixed in stage two.\n",
        encoding="utf-8",
    )

    manager.rebuild_search_index()

    working_set_hits = manager.search_sessions("rehydration", limit=5)
    transcript_hits = manager.search_sessions("cursor drift", limit=5)

    assert working_set_hits
    assert working_set_hits[0]["session_id"] == session.id
    assert "rehydration" in working_set_hits[0]["snippet"].lower()

    assert transcript_hits
    assert transcript_hits[0]["session_id"] == session.id
    assert "cursor drift" in transcript_hits[0]["snippet"].lower()


def test_read_session_supports_working_set_and_transcript_modes(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-read-modes")
    session.title = "Read modes"
    session.add_message("user", "hello")
    manager.save(session)
    assert session.id is not None

    session_dir = _conversation_dir(tmp_path, "cli:test-read-modes") / "sessions" / session.id
    working_set_text = "# Working Set\n\nCurrent goal: verify read modes.\n"
    transcript_text = "# Transcript\n\nUSER: hello\n"
    (session_dir / "working_set.md").write_text(working_set_text, encoding="utf-8")
    (session_dir / "transcript.md").write_text(transcript_text, encoding="utf-8")

    working_set_result = manager.read_session(session.id, mode="working_set")
    transcript_result = manager.read_session(session.id, mode="transcript")

    assert working_set_result["content"] == working_set_text
    assert transcript_result["content"] == transcript_text
