"""File-backed session artifacts for human-readable local inspection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanobot.utils.helpers import ensure_dir, get_data_path, safe_filename

if TYPE_CHECKING:
    from nanobot.session.manager import Session


@dataclass(frozen=True)
class SessionArtifactPaths:
    """Resolved artifact paths for one session."""

    conversation_dir: Path
    state_file: Path
    session_dir: Path
    meta_file: Path
    events_file: Path
    summary_file: Path
    working_set_file: Path
    transcript_file: Path


class SessionArtifactStore:
    """Mirror sessions into a file-first artifact layout."""

    _WORKING_SET_AUTO_MARKER = "<!-- nanobot:auto-working-set -->"
    _SUMMARY_CHECKPOINT_TURNS = 10
    _SUMMARY_AUTO_SOURCE = "auto"
    _SUMMARY_MANUAL_SOURCE = "manual"
    _SUMMARY_AUTO_MARKER = "<!-- nanobot:auto-summary -->"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.root = ensure_dir(get_data_path() / "conversations")

    @staticmethod
    def conversation_slug(conversation_key: str) -> str:
        normalized = conversation_key.replace(":", "__")
        return safe_filename(normalized)

    def paths_for(self, conversation_key: str, session_id: str) -> SessionArtifactPaths:
        conversation_dir = ensure_dir(self.root / self.conversation_slug(conversation_key))
        state_file = conversation_dir / "state.toml"
        session_dir = ensure_dir(conversation_dir / "sessions" / safe_filename(session_id))
        return SessionArtifactPaths(
            conversation_dir=conversation_dir,
            state_file=state_file,
            session_dir=session_dir,
            meta_file=session_dir / "meta.toml",
            events_file=session_dir / "events.jsonl",
            summary_file=session_dir / "summary.md",
            working_set_file=session_dir / "working_set.md",
            transcript_file=session_dir / "transcript.md",
        )

    def write_conversation_state(self, snapshot: dict[str, Any]) -> None:
        """Write conversation-level state."""
        conversation = snapshot.get("conversation") or {}
        conversation_key = str(conversation.get("conversation_key") or conversation.get("key") or "").strip()
        if not conversation_key:
            return

        active_session_id = str(snapshot.get("active_session_id") or "").strip()
        paths = self.paths_for(conversation_key, active_session_id or "_state")
        sessions = snapshot.get("sessions") or []

        payload = {
            "conversation_key": conversation_key,
            "conversation_id": conversation.get("id"),
            "channel": conversation.get("channel"),
            "chat_id": conversation.get("chat_id"),
            "active_session_id": active_session_id,
            "updated_at": conversation.get("updated_at"),
            "sessions": [
                {
                    "id": item.get("id"),
                    "key": item.get("key"),
                    "title": item.get("title"),
                    "summary": item.get("summary"),
                    "updated_at": item.get("updated_at"),
                }
                for item in sessions
                if isinstance(item, dict)
            ],
        }
        paths.state_file.write_text(self._to_toml(payload), encoding="utf-8")

    def write_session(self, session: "Session") -> SessionArtifactPaths | None:
        """Write file artifacts for a single session."""
        if not session.id or not session.conversation_key:
            return None

        paths = self.paths_for(session.conversation_key, session.id)
        meta_payload = {
            "id": session.id,
            "key": session.key,
            "conversation_key": session.conversation_key,
            "conversation_id": session.conversation_id,
            "title": session.title,
            "summary": session.summary,
            "kind": session.kind,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "last_consolidated": session.last_consolidated,
            "metadata": session.metadata,
        }
        paths.meta_file.write_text(self._to_toml(meta_payload), encoding="utf-8")

        event_lines = []
        for idx, message in enumerate(session.messages):
            payload = {"index": idx, **message}
            event_lines.append(json.dumps(payload, ensure_ascii=False))
        paths.events_file.write_text("\n".join(event_lines) + ("\n" if event_lines else ""), encoding="utf-8")

        paths.summary_file.write_text(self._build_summary_markdown(session), encoding="utf-8")
        paths.transcript_file.write_text(self._build_transcript_markdown(session), encoding="utf-8")
        self._ensure_working_set(paths, session)
        return paths

    def refresh_summary_checkpoint(self, session: "Session") -> None:
        """Refresh the session summary only at checkpoint turns."""
        metadata = session.metadata if isinstance(session.metadata, dict) else {}
        session.metadata = metadata

        current_summary = session.summary.strip()
        stored_auto_text = str(metadata.get("summary_auto_text") or "").strip()
        summary_source = str(metadata.get("summary_source") or "").strip()
        last_checkpoint_turn = int(metadata.get("summary_checkpoint_turn") or 0)
        user_turns = self._count_user_turns(session)
        checkpoint_turn = (user_turns // self._SUMMARY_CHECKPOINT_TURNS) * self._SUMMARY_CHECKPOINT_TURNS

        if current_summary:
            if summary_source == self._SUMMARY_AUTO_SOURCE and stored_auto_text and current_summary != stored_auto_text:
                metadata["summary_source"] = self._SUMMARY_MANUAL_SOURCE
                metadata.pop("summary_auto_text", None)
                return
            if summary_source != self._SUMMARY_AUTO_SOURCE:
                metadata["summary_source"] = self._SUMMARY_MANUAL_SOURCE
                metadata.pop("summary_auto_text", None)
                return

        if checkpoint_turn < self._SUMMARY_CHECKPOINT_TURNS:
            return
        if metadata.get("summary_source") == self._SUMMARY_MANUAL_SOURCE and current_summary:
            return
        if checkpoint_turn <= last_checkpoint_turn and current_summary:
            return

        previous_summary = stored_auto_text if summary_source == self._SUMMARY_AUTO_SOURCE else current_summary
        new_summary = self._build_checkpoint_summary_text(
            session,
            checkpoint_turn=checkpoint_turn,
            previous_summary=previous_summary,
        )
        session.summary = new_summary
        metadata["summary_source"] = self._SUMMARY_AUTO_SOURCE
        metadata["summary_checkpoint_turn"] = checkpoint_turn
        metadata["summary_auto_text"] = new_summary

    def load_working_set(self, session: "Session") -> tuple[Path | None, str]:
        """Return the current working-set file path and content for a session."""
        if not session.id or not session.conversation_key:
            return None, ""

        paths = self.paths_for(session.conversation_key, session.id)
        self._ensure_working_set(paths, session)
        if not paths.working_set_file.exists():
            return paths.working_set_file, ""
        return paths.working_set_file, paths.working_set_file.read_text(encoding="utf-8").strip()

    def _build_summary_markdown(self, session: "Session") -> str:
        title = session.title.strip() or f"Session {session.id}"
        summary = session.summary.strip()
        if not summary:
            summary = self._pending_summary_text(session)

        lines = [
            f"# {title}",
            "",
            f"- Session ID: {session.id}",
            f"- Session Key: {session.key}",
            f"- Conversation: {session.conversation_key or '(none)'}",
            f"- Kind: {session.kind}",
            f"- Updated At: {session.updated_at.isoformat()}",
        ]
        app_server_thread_id = str(session.metadata.get("app_server_thread_id") or "").strip()
        if app_server_thread_id:
            lines.append(f"- App Server Thread: {app_server_thread_id}")
        if session.summary.strip() and str(session.metadata.get("summary_source") or "") == self._SUMMARY_AUTO_SOURCE:
            checkpoint_turn = int(session.metadata.get("summary_checkpoint_turn") or 0)
            if checkpoint_turn:
                lines.append(f"- Summary Checkpoint: {checkpoint_turn} user turns")
        lines.extend([
            "",
            "## Summary",
            "",
            self._SUMMARY_AUTO_MARKER if str(session.metadata.get("summary_source") or "") == self._SUMMARY_AUTO_SOURCE else "",
            "",
            summary or "(empty)",
            "",
        ])
        return "\n".join(lines)

    @staticmethod
    def _latest_summary_fallback(session: "Session") -> str:
        for message in reversed(session.messages):
            role = str(message.get("role") or "").lower()
            if role != "assistant":
                continue
            content = str(message.get("content") or "").strip()
            if content:
                return content

        for message in reversed(session.messages):
            content = str(message.get("content") or "").strip()
            if content:
                return content
        return "(empty)"

    def _ensure_working_set(self, paths: SessionArtifactPaths, session: "Session") -> None:
        """Create an editable working-set scaffold without clobbering manual edits."""
        if paths.working_set_file.exists():
            existing = paths.working_set_file.read_text(encoding="utf-8").strip()
            if existing and self._WORKING_SET_AUTO_MARKER not in existing:
                return
        paths.working_set_file.write_text(self._build_working_set_markdown(session), encoding="utf-8")

    def _build_transcript_markdown(self, session: "Session") -> str:
        title = session.title.strip() or f"Session {session.id}"
        lines = [
            f"# Transcript: {title}",
            "",
            f"- Session ID: {session.id or '(unsaved)'}",
            f"- Session Key: {session.key}",
            f"- Conversation: {session.conversation_key or '(none)'}",
            f"- Updated At: {session.updated_at.isoformat()}",
            "",
        ]
        if not session.messages:
            lines.extend(["(empty)", ""])
            return "\n".join(lines)

        for message in session.messages:
            role = str(message.get("role") or "unknown").strip().lower() or "unknown"
            heading = role.upper()
            if role == "tool":
                tool_name = str(message.get("name") or message.get("tool_call_id") or "tool").strip()
                if tool_name:
                    heading = f"TOOL {tool_name}"
            timestamp = str(message.get("timestamp") or "").strip()
            content = str(message.get("content") or "").strip() or "(empty)"
            lines.append(f"## {heading}")
            if timestamp:
                lines.append(f"- Timestamp: {timestamp}")
            tools_used = message.get("tools_used")
            if isinstance(tools_used, list) and tools_used:
                cleaned = [str(tool).strip() for tool in tools_used if str(tool).strip()]
                if cleaned:
                    lines.append(f"- Tools Used: {', '.join(cleaned)}")
            lines.extend(["", content, ""])
        return "\n".join(lines).rstrip() + "\n"

    def _build_working_set_markdown(self, session: "Session") -> str:
        current_goal = self._latest_message_text(session, role="user")
        if not current_goal:
            current_goal = session.title.strip() or session.summary.strip() or "(not set yet)"

        summary = session.summary.strip() or self._latest_summary_fallback(session)
        latest_assistant = self._latest_message_text(session, role="assistant") or "(none yet)"
        recent_tools = self._latest_tools_used(session)

        lines = [
            "# Working Set",
            "",
            self._WORKING_SET_AUTO_MARKER,
            "",
            "This file tracks the immediate execution state for this session.",
            "Keep it short, concrete, and easy to update during long-running work.",
            "",
            "## Current Goal",
            "",
            current_goal,
            "",
            "## Session Summary",
            "",
            summary,
            "",
            "## Latest Assistant State",
            "",
            latest_assistant,
            "",
            "## Active Files",
            "",
            "- (add relevant files here when they become important)",
            "",
            "## Open TODOs",
            "",
            "- Continue from the latest unfinished request.",
            "",
            "## Constraints",
            "",
            "- Preserve immediate working state across compaction or thread restarts.",
            "- Update this file when the active goal, blockers, or next actions change.",
        ]
        if recent_tools:
            lines.extend([
                "",
                "## Recent Tools",
                "",
                *[f"- `{tool}`" for tool in recent_tools],
            ])
        return "\n".join(lines).rstrip() + "\n"

    def _pending_summary_text(self, session: "Session") -> str:
        user_turns = self._count_user_turns(session)
        next_checkpoint = self._SUMMARY_CHECKPOINT_TURNS
        if user_turns >= self._SUMMARY_CHECKPOINT_TURNS:
            next_checkpoint = ((user_turns // self._SUMMARY_CHECKPOINT_TURNS) + 1) * self._SUMMARY_CHECKPOINT_TURNS
        return (
            "Automatic session summaries update every "
            f"{self._SUMMARY_CHECKPOINT_TURNS} user turns.\n\n"
            f"Current user turns: {user_turns}\n"
            f"Next summary checkpoint: {next_checkpoint}"
        )

    def _build_checkpoint_summary_text(
        self,
        session: "Session",
        *,
        checkpoint_turn: int,
        previous_summary: str,
    ) -> str:
        recent_turns = self._select_recent_turn_pairs(session, count=self._SUMMARY_CHECKPOINT_TURNS)
        lines = [
            f"Automatic checkpoint summary after {checkpoint_turn} user turns.",
            "",
        ]
        if previous_summary:
            lines.extend([
                "Previous checkpoint context:",
                previous_summary,
                "",
            ])
        if session.title.strip():
            lines.extend([
                f"Session title: {session.title.strip()}",
                "",
            ])
        lines.append("Recent turn checkpoints:")
        for idx, (user_text, assistant_text) in enumerate(recent_turns, start=1):
            lines.append(f"{idx}. User: {user_text}")
            if assistant_text:
                lines.append(f"   Assistant: {assistant_text}")
        return "\n".join(lines).strip()

    @staticmethod
    def _count_user_turns(session: "Session") -> int:
        return sum(1 for message in session.messages if str(message.get("role") or "").lower() == "user")

    def _select_recent_turn_pairs(self, session: "Session", *, count: int) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        current_user = ""
        current_assistant = ""

        for message in session.messages:
            role = str(message.get("role") or "").lower()
            content = " ".join(str(message.get("content") or "").strip().split())
            if not content:
                continue
            if role == "user":
                if current_user:
                    pairs.append((current_user, current_assistant))
                    current_assistant = ""
                current_user = self._truncate_summary_text(content, limit=220)
            elif role == "assistant" and current_user and not current_assistant:
                current_assistant = self._truncate_summary_text(content, limit=260)

        if current_user:
            pairs.append((current_user, current_assistant))
        return pairs[-count:]

    @staticmethod
    def _truncate_summary_text(text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _latest_message_text(session: "Session", *, role: str) -> str:
        for message in reversed(session.messages):
            if str(message.get("role") or "").lower() != role:
                continue
            content = " ".join(str(message.get("content") or "").strip().split())
            if content:
                if len(content) > 800:
                    return content[:797] + "..."
                return content
        return ""

    @staticmethod
    def _latest_tools_used(session: "Session") -> list[str]:
        for message in reversed(session.messages):
            if str(message.get("role") or "").lower() != "assistant":
                continue
            tools = message.get("tools_used")
            if not isinstance(tools, list):
                continue
            cleaned = [str(tool).strip() for tool in tools if str(tool).strip()]
            if cleaned:
                return cleaned
        return []

    def _to_toml(self, payload: dict[str, Any]) -> str:
        lines: list[str] = []
        self._append_toml_table(lines, payload)
        return "\n".join(lines).rstrip() + "\n"

    def _append_toml_table(
        self,
        lines: list[str],
        payload: dict[str, Any],
        prefix: str | None = None,
    ) -> None:
        scalar_items: list[tuple[str, Any]] = []
        nested_items: list[tuple[str, dict[str, Any]]] = []

        for key, value in payload.items():
            if isinstance(value, dict):
                nested_items.append((key, value))
            else:
                scalar_items.append((key, value))

        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalar_items:
            lines.append(f"{key} = {self._toml_value(value)}")
        if scalar_items and nested_items:
            lines.append("")

        for idx, (key, value) in enumerate(nested_items):
            section = key if not prefix else f"{prefix}.{key}"
            self._append_toml_table(lines, value, prefix=section)
            if idx != len(nested_items) - 1:
                lines.append("")

    def _toml_value(self, value: Any) -> str:
        if value is None:
            return '""'
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return "[" + ", ".join(self._toml_value(item) for item in value) + "]"
        if isinstance(value, dict):
            parts = [f"{key} = {self._toml_value(item)}" for key, item in value.items()]
            return "{ " + ", ".join(parts) + " }"
        return json.dumps(value, ensure_ascii=False)
