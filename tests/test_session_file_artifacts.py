from __future__ import annotations

import json
from pathlib import Path

from nanobot.session.file_artifacts import SessionArtifactStore
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import get_data_path


def _conversation_dir(root: Path, conversation_key: str) -> Path:
    return get_data_path() / "conversations" / conversation_key.replace(":", "__")


def test_session_manager_writes_file_artifacts_for_saved_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-artifacts")
    session.title = "Artifact Session"
    session.summary = "Short summary for file artifacts."
    session.metadata["app_server_thread_id"] = "thread-123"
    session.metadata["app_server_last_turn_id"] = "turn-456"
    session.add_message("user", "hello")
    session.add_message("assistant", "world", tools_used=["sessions"])
    manager.save(session)

    assert session.id is not None
    conversation_dir = _conversation_dir(tmp_path, "cli:test-artifacts")
    session_dir = conversation_dir / "sessions" / session.id

    state_text = (conversation_dir / "state.toml").read_text(encoding="utf-8")
    meta_text = (session_dir / "meta.toml").read_text(encoding="utf-8")
    summary_text = (session_dir / "summary.md").read_text(encoding="utf-8")
    working_set_text = (session_dir / "working_set.md").read_text(encoding="utf-8")
    transcript_text = (session_dir / "transcript.md").read_text(encoding="utf-8")
    event_lines = (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()

    assert 'active_session_id = "{}"'.format(session.id) in state_text
    assert 'conversation_key = "cli:test-artifacts"' in state_text

    assert 'id = "{}"'.format(session.id) in meta_text
    assert 'app_server_thread_id = "thread-123"' in meta_text
    assert 'app_server_last_turn_id = "turn-456"' in meta_text
    assert 'title = "Artifact Session"' in meta_text

    assert "# Artifact Session" in summary_text
    assert "Short summary for file artifacts." in summary_text
    assert "App Server Thread: thread-123" in summary_text
    assert "# Working Set" in working_set_text
    assert "## Current Goal" in working_set_text
    assert "hello" in working_set_text
    assert "world" in working_set_text
    assert "`sessions`" in working_set_text
    assert "# Transcript: Artifact Session" in transcript_text
    assert "## USER" in transcript_text
    assert "## ASSISTANT" in transcript_text
    assert "hello" in transcript_text
    assert "world" in transcript_text

    events = [json.loads(line) for line in event_lines]
    assert [event["role"] for event in events] == ["user", "assistant"]
    assert events[1]["tools_used"] == ["sessions"]


def test_conversation_state_file_tracks_active_session_switch(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    first = manager.get_active_session("cli:test-switch")
    manager.save(first)
    assert first.id is not None

    created = manager.create_session("cli:test-switch", title="Second", switch_to=True)
    manager.switch_session("cli:test-switch", first.id)

    conversation_dir = _conversation_dir(tmp_path, "cli:test-switch")
    state_text = (conversation_dir / "state.toml").read_text(encoding="utf-8")

    assert 'active_session_id = "{}"'.format(first.id) in state_text
    assert created["id"] in state_text

    created_session_dir = conversation_dir / "sessions" / created["id"]
    assert (created_session_dir / "meta.toml").exists()
    assert (created_session_dir / "events.jsonl").exists()
    assert (created_session_dir / "summary.md").exists()
    assert (created_session_dir / "working_set.md").exists()
    assert (created_session_dir / "transcript.md").exists()


def test_working_set_preserves_manual_edits(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-working-set")
    session.title = "Working set"
    session.add_message("user", "initial request")
    manager.save(session)
    assert session.id is not None

    session_dir = _conversation_dir(tmp_path, "cli:test-working-set") / "sessions" / session.id
    working_set_path = session_dir / "working_set.md"
    manual_text = "# Working Set\n\nManual note that should survive.\n"
    working_set_path.write_text(manual_text, encoding="utf-8")

    session.add_message("assistant", "follow-up response")
    manager.save(session)

    assert working_set_path.read_text(encoding="utf-8") == manual_text


def test_session_artifact_store_does_not_migrate_legacy_workspace_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    legacy_root = workspace / ".nanobot" / "conversations" / "cli__legacy"
    legacy_root.mkdir(parents=True)
    (legacy_root / "state.toml").write_text('conversation_key = "cli:legacy"\n', encoding="utf-8")

    store = SessionArtifactStore(workspace)

    assert store.root == get_data_path() / "conversations"
    assert legacy_root.exists()
    assert not any(store.root.iterdir())


def test_summary_stays_pending_before_tenth_user_turn(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-summary-pending")
    session.title = "Pending summary"
    for idx in range(3):
        session.add_message("user", f"user turn {idx + 1}")
        session.add_message("assistant", f"assistant turn {idx + 1}")
    manager.save(session)
    assert session.id is not None

    summary_text = (
        _conversation_dir(tmp_path, "cli:test-summary-pending") / "sessions" / session.id / "summary.md"
    ).read_text(encoding="utf-8")

    assert "Automatic session summaries update every 10 user turns." in summary_text
    assert "Current user turns: 3" in summary_text
    assert "assistant turn 3" not in summary_text
    assert session.summary == ""


def test_summary_updates_only_on_tenth_user_turn_checkpoint(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_active_session("cli:test-summary-checkpoint")
    session.title = "Checkpoint summary"
    for idx in range(10):
        session.add_message("user", f"user turn {idx + 1}")
        session.add_message("assistant", f"assistant turn {idx + 1}")
    manager.save(session)
    assert session.id is not None

    summary_path = _conversation_dir(tmp_path, "cli:test-summary-checkpoint") / "sessions" / session.id / "summary.md"
    summary_after_ten = summary_path.read_text(encoding="utf-8")

    assert "Automatic checkpoint summary after 10 user turns." in summary_after_ten
    assert "Summary Checkpoint: 10 user turns" in summary_after_ten
    assert "10. User: user turn 10" in summary_after_ten
    assert session.metadata["summary_checkpoint_turn"] == 10
    assert session.metadata["summary_source"] == "auto"

    session.add_message("user", "user turn 11")
    session.add_message("assistant", "assistant turn 11")
    manager.save(session)

    summary_after_eleven = summary_path.read_text(encoding="utf-8")
    assert "Automatic checkpoint summary after 10 user turns." in summary_after_eleven
    assert "Summary Checkpoint: 10 user turns" in summary_after_eleven
    assert "user turn 11" not in summary_after_eleven
