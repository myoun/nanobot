"""Agent loop: the core processing engine."""

import asyncio
from contextlib import AsyncExitStack
import json
import json_repair
import os
from pathlib import Path
import re
import shlex
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.complete import CompleteTaskTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.report import ReportToUserTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.security.approval_store import ApprovalStore
from nanobot.security.privileged_client import PrivilegedClient
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import get_data_path

if TYPE_CHECKING:
    from nanobot.config.schema import ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _COMPLETE_TOOL_NAME = "complete_task"
    _REPORT_TOOL_NAME = "report_to_user"
    _REQUEST_MODE_DO = "DO"
    _REQUEST_MODE_CHAT = "CHAT"
    _NON_PROGRESS_TOOLS = {_COMPLETE_TOOL_NAME, _REPORT_TOOL_NAME}
    _MODE_CLASSIFIER_MAX_HISTORY = 10
    _MODE_CLASSIFIER_ITEM_MAX_CHARS = 220
    _DO_MODE_NO_TOOL_RESPONSE = (
        "This request requires real tool execution, but no task-execution tools are currently available. "
        "I can provide a patch/command plan only; runtime execution must be enabled first."
    )
    _NO_ACTION_NUDGE = (
        "Continue working on this request. "
        "Use tools if needed. Call complete_task only when fully done, "
        "including required fields: final_answer/artifacts/evidence/actions_taken."
    )
    _ACTION_REQUEST_NUDGE = (
        "This request appears to require tool execution. "
        "Do not claim completion before executing and verifying with at least one relevant tool successfully. "
        "Then call complete_task with required fields "
        "(final_answer, artifacts, evidence, actions_taken)."
    )
    _ACTION_RETRY_REASON_NO_PROGRESS = (
        "Retry reason: external tool attempts have not produced verified progress yet. "
        "Try another relevant tool call, or if blocked, call complete_task(final_answer=...) "
        "with a concise failure reason and the exact user action required."
    )
    _ACTION_RETRY_REASON_REPORT_ONLY = (
        "Retry reason: report_to_user is for intermediate updates only and does not count as task execution. "
        "Run at least one task-executing tool (e.g., read_file/write_file/edit_file/list_dir/exec/web_fetch/web_search/message with real delivery), "
        "then call complete_task with required fields final_answer/artifacts/evidence/actions_taken."
    )
    _ACTION_RETRY_REASON_MISSING_EVIDENCE = (
        "Retry reason: complete_task in TASK_REQUEST mode requires non-empty `evidence` and `actions_taken`. "
        "Include concrete execution evidence (command/tool outputs) and real actions performed."
    )
    _ACTION_RETRY_REASON_INVALID_COMPLETE_PAYLOAD = (
        "Retry reason: complete_task payload is invalid. "
        "Provide required fields: `final_answer`, `artifacts`, `evidence`, `actions_taken`."
    )
    _COMPLETION_REJECT_NUDGE = (
        "Your complete_task call was rejected. "
        "Keep working and call complete_task only after verified progress "
        "with required fields final_answer/artifacts/evidence/actions_taken."
    )
    _MAX_NO_TOOL_TEXT_ROUNDS = 3
    _MAX_NO_TOOL_EMPTY_ROUNDS = 4
    _PREFILL_FILE_CANDIDATES = ("workspace/PREFILL.md", "PREFILL.md")
    _SESSION_TITLE_MAX_CHARS = 60
    _SESSION_TITLE_CONTEXT_MESSAGES = 6
    _SESSION_TITLE_MIN_USER_MESSAGES = 2
    _SESSION_TITLE_MIN_ASSISTANT_MESSAGES = 1
    _SESSION_TITLE_RETRY_MESSAGE_GAP = 3
    _SESSION_TITLE_MAX_AUTO_ATTEMPTS = 3
    _SESSION_DEFAULT_TITLE = "New chat"
    _NO_TOOL_FALLBACK = (
        "I couldn't make progress with tool execution or completion signaling. "
        "Please provide a more specific next instruction."
    )
    _AGENT_BROWSER_AUTO_CLOSE_CMD = "agent-browser close >/dev/null 2>&1 || true"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 30,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig

        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        try:
            approvals_path = get_data_path() / "approvals" / "requests.json"
            self._approval_store = ApprovalStore(
                approvals_path,
                ttl_seconds=self.exec_config.approval_ttl_sec,
                single_pending_per_chat=self.exec_config.single_pending_per_chat,
            )
        except OSError:
            approvals_path = Path.home() / ".nanobot" / "approvals" / "requests.json"
            fallback_path = self.workspace / ".nanobot" / "approvals" / "requests.json"
            logger.warning(
                f"Approval store path not writable ({approvals_path}); using workspace fallback: {fallback_path}"
            )
            self._approval_store = ApprovalStore(
                fallback_path,
                ttl_seconds=self.exec_config.approval_ttl_sec,
                single_pending_per_chat=self.exec_config.single_pending_per_chat,
            )
        self._privileged_client: PrivilegedClient | None = None
        if self.exec_config.privileged_enabled:
            if os.name == "posix":
                self._privileged_client = PrivilegedClient(self.exec_config.privileged_socket)
            else:
                logger.warning(
                    "Privileged execution is enabled in config but unsupported on non-Unix runtime; ignoring."
                )
        self._register_default_tools()
        self._process_lock = asyncio.Lock()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                privileged_enabled=self.exec_config.privileged_enabled,
                approval_store=self._approval_store,
            )
        )

        # Web tools
        # OpenAI Codex provider has native web search via Responses API tools.
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        if not isinstance(self.provider, OpenAICodexProvider):
            self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(CompleteTaskTool())

        # Progress-report tool (text updates to current chat)
        report_tool = ReportToUserTool(send_callback=self.bus.publish_outbound)
        self.tools.register(report_tool)

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers

        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str, sender_id: str = "") -> None:
        """Update context for all tools that need routing info."""
        if exec_tool := self.tools.get("exec"):
            if isinstance(exec_tool, ExecTool):
                exec_tool.set_context(channel, chat_id, sender_id)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if report_tool := self.tools.get("report_to_user"):
            if isinstance(report_tool, ReportToUserTool):
                report_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _truncate_preview(text: str, max_len: int = 800) -> str:
        clean = text.strip()
        if len(clean) <= max_len:
            return clean
        return clean[:max_len] + f"\n... (truncated, {len(clean) - max_len} more chars)"

    @staticmethod
    def _conversation_key(channel: str, chat_id: str) -> str:
        return f"{channel}:{chat_id}"

    @staticmethod
    def _extract_requested_session_id(metadata: dict[str, Any] | None) -> str | None:
        if not isinstance(metadata, dict):
            return None

        direct = metadata.get("session_id")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        session_block = metadata.get("session")
        if isinstance(session_block, dict):
            sid = session_block.get("id")
            if isinstance(sid, str) and sid.strip():
                return sid.strip()

        web_block = metadata.get("web")
        if isinstance(web_block, dict):
            sid = web_block.get("session_id")
            if isinstance(sid, str) and sid.strip():
                return sid.strip()

        nanobot_block = metadata.get("_nanobot")
        if isinstance(nanobot_block, dict):
            sid = nanobot_block.get("session_id")
            if isinstance(sid, str) and sid.strip():
                return sid.strip()

        return None

    @staticmethod
    def _parse_command_tokens(raw_command: str) -> list[str]:
        text = raw_command.strip()
        if not text:
            return []
        try:
            return shlex.split(text)
        except ValueError:
            return text.split()

    @staticmethod
    def _session_state_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {"session_state": snapshot}

    @staticmethod
    def _resolve_session_token(snapshot: dict[str, Any], token: str) -> str | None:
        sessions = snapshot.get("sessions")
        if not isinstance(sessions, list):
            return None

        cleaned = token.strip()
        if not cleaned:
            return None

        if cleaned in {"active", "current"}:
            active = snapshot.get("active_session_id")
            return str(active) if isinstance(active, str) and active else None

        for item in sessions:
            if not isinstance(item, dict):
                continue
            if str(item.get("id", "")) == cleaned:
                return cleaned

        if cleaned.isdigit():
            idx = int(cleaned) - 1
            if 0 <= idx < len(sessions):
                item = sessions[idx]
                if isinstance(item, dict):
                    sid = item.get("id")
                    if isinstance(sid, str) and sid:
                        return sid

        return None

    def _render_session_list_text(self, snapshot: dict[str, Any]) -> str:
        sessions = snapshot.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            return "No sessions in this chat yet. Use /new to create one."

        lines = ["Sessions in this chat:"]
        for idx, item in enumerate(sessions, start=1):
            if not isinstance(item, dict):
                continue
            marker = "*" if bool(item.get("active")) else " "
            pinned = "📌 " if bool(item.get("pinned", False)) else ""
            title = str(item.get("title") or self._SESSION_DEFAULT_TITLE)
            sid = str(item.get("id") or "")
            lines.append(f"{idx}. [{marker}] {pinned}{title} ({sid})")

        lines.append(
            "Use /new, /session switch <id|index>, /session rename <id|index|active> <title>, /session pin <id|index>, /session unpin <id|index>, /session search <keyword>, /session delete <id|index|active>."
        )
        return "\n".join(lines)

    def _handle_session_command(
        self,
        *,
        msg: InboundMessage,
        cmd_name: str,
        conversation_key: str,
    ) -> OutboundMessage | None:
        if cmd_name not in {"/new", "/sessions", "/session"}:
            return None

        if cmd_name == "/new":
            created = self.sessions.create_session(conversation_key, title=None, switch_to=True)
            snapshot = self.sessions.list_conversation_sessions(conversation_key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    f"Started a new session: {created['title']} ({created['id']}).\n"
                    "Use /sessions to browse or switch sessions."
                ),
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(snapshot),
                ),
            )

        if cmd_name == "/sessions":
            snapshot = self.sessions.list_conversation_sessions(conversation_key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self._render_session_list_text(snapshot),
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(snapshot),
                ),
            )

        tokens = self._parse_command_tokens(msg.content)
        if len(tokens) < 2:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "Session commands:\n"
                    "/session list\n"
                    "/session current\n"
                    "/session new [title]\n"
                    "/session switch <id|index|active>\n"
                    "/session rename <id|index|active> <new title>\n"
                    "/session pin <id|index|active>\n"
                    "/session unpin <id|index|active>\n"
                    "/session search <keyword>\n"
                    "/session delete <id|index|active>"
                ),
            )

        action = tokens[1].lower()
        snapshot = self.sessions.list_conversation_sessions(conversation_key)

        if action in {"list", "ls"}:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self._render_session_list_text(snapshot),
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(snapshot),
                ),
            )

        if action in {"current", "active"}:
            sid = str(snapshot.get("active_session_id") or "")
            current = None
            for item in snapshot.get("sessions", []):
                if isinstance(item, dict) and str(item.get("id", "")) == sid:
                    current = item
                    break
            if current is None:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="No active session.",
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Current session: {current.get('title')} ({current.get('id')})",
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(snapshot),
                ),
            )

        if action in {"new", "create"}:
            title = " ".join(tokens[2:]).strip() or None
            created = self.sessions.create_session(conversation_key, title=title, switch_to=True)
            snapshot = self.sessions.list_conversation_sessions(conversation_key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Switched to new session: {created['title']} ({created['id']}).",
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(snapshot),
                ),
            )

        if action in {"pin", "unpin"}:
            if len(tokens) < 3:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Usage: /session {action} <id|index|active>",
                )
            target = self._resolve_session_token(snapshot, tokens[2])
            if not target:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Unknown session target. Run /sessions to check ids.",
                )
            pinned = action == "pin"
            updated = self.sessions.set_session_pinned(
                conversation_key,
                target,
                pinned=pinned,
            )
            snapshot = self.sessions.list_conversation_sessions(conversation_key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    f"{'Pinned' if pinned else 'Unpinned'} session: "
                    f"{updated['title']} ({updated['id']})."
                ),
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(snapshot),
                ),
            )

        if action in {"search", "find"}:
            keyword = " ".join(tokens[2:]).strip() if len(tokens) > 2 else ""
            if not keyword:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /session search <keyword>",
                )
            found = self.sessions.search_sessions(conversation_key, keyword)
            found_sessions = found.get("sessions")
            if not isinstance(found_sessions, list) or not found_sessions:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"No sessions matched '{keyword}'.",
                    metadata=self._merge_outbound_metadata(
                        msg.metadata,
                        self._session_state_metadata(
                            self.sessions.list_conversation_sessions(conversation_key)
                        ),
                    ),
                )
            lines = [f"Matches for '{keyword}':"]
            for idx, item in enumerate(found_sessions, start=1):
                if not isinstance(item, dict):
                    continue
                marker = "*" if bool(item.get("active")) else " "
                pin = "📌 " if bool(item.get("pinned", False)) else ""
                lines.append(f"{idx}. [{marker}] {pin}{item.get('title')} ({item.get('id')})")
            lines.append("Use /session switch <id> to move to one result.")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="\n".join(lines),
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(
                        self.sessions.list_conversation_sessions(conversation_key)
                    ),
                ),
            )

        if action in {"switch", "use"}:
            if len(tokens) < 3:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /session switch <id|index|active>",
                )
            target = self._resolve_session_token(snapshot, tokens[2])
            if not target:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Unknown session target. Run /sessions to check ids.",
                )
            switched = self.sessions.switch_session(conversation_key, target)
            snapshot = self.sessions.list_conversation_sessions(conversation_key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Switched to session: {switched['title']} ({switched['id']}).",
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(snapshot),
                ),
            )

        if action == "rename":
            if len(tokens) < 4:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /session rename <id|index|active> <new title>",
                )
            target = self._resolve_session_token(snapshot, tokens[2])
            if not target:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Unknown session target. Run /sessions to check ids.",
                )
            new_title = " ".join(tokens[3:]).strip()
            if not new_title:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="New title must not be empty.",
                )
            renamed = self.sessions.rename_session(
                conversation_key,
                target,
                new_title,
                auto_title=False,
            )
            snapshot = self.sessions.list_conversation_sessions(conversation_key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Renamed session to: {renamed['title']} ({renamed['id']}).",
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(snapshot),
                ),
            )

        if action in {"delete", "remove", "rm"}:
            if len(tokens) < 3:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /session delete <id|index|active>",
                )
            target = self._resolve_session_token(snapshot, tokens[2])
            if not target:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Unknown session target. Run /sessions to check ids.",
                )
            result = self.sessions.delete_session(conversation_key, target)
            after_snapshot = self.sessions.list_conversation_sessions(conversation_key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    f"Deleted session ({result['deleted_session_id']}). "
                    f"Active session is now {result['active_session_id']}."
                ),
                metadata=self._merge_outbound_metadata(
                    msg.metadata,
                    self._session_state_metadata(after_snapshot),
                ),
            )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=(
                "Unknown /session command. Use one of: list, current, new, switch, rename, delete."
            ),
        )

    @classmethod
    def _normalize_generated_title(cls, raw_title: str) -> str:
        text = (raw_title or "").strip()
        if not text:
            return ""

        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if "```" in text:
                text = text.rsplit("```", 1)[0]
            text = text.strip()

        text = text.splitlines()[0].strip()
        text = re.sub(r"^(?:title\s*[:\-]|\-|\*|\d+\.)\s*", "", text, flags=re.IGNORECASE)
        text = text.strip("\"'` ")
        text = " ".join(text.split())
        return text[: cls._SESSION_TITLE_MAX_CHARS]

    @classmethod
    def _is_low_quality_generated_title(cls, title: str) -> bool:
        normalized = " ".join((title or "").lower().split())
        if not normalized:
            return True
        blocked = {
            "new chat",
            "chat",
            "conversation",
            "session",
            "untitled",
        }
        if normalized in blocked:
            return True
        if len(normalized) < 4:
            return True
        words = normalized.split()
        if len(words) == 1 and len(words[0]) < 8:
            return True
        return False

    async def _maybe_generate_session_title(
        self,
        *,
        conversation_key: str,
        session_id: str,
        session: Session,
    ) -> str | None:
        snapshot = self.sessions.list_conversation_sessions(conversation_key)
        target: dict[str, Any] | None = None
        for item in snapshot.get("sessions", []):
            if isinstance(item, dict) and str(item.get("id", "")) == session_id:
                target = item
                break
        if target is None:
            return None
        if not bool(target.get("auto_title", True)):
            return None
        if bool(target.get("title_locked", False)):
            return None

        attempts = int(target.get("title_auto_attempts", 0) or 0)
        if attempts >= self._SESSION_TITLE_MAX_AUTO_ATTEMPTS:
            return None

        turns: list[dict[str, Any]] = []
        user_count = 0
        assistant_count = 0
        for msg in session.messages:
            role = str(msg.get("role") or "").strip()
            if role not in {"user", "assistant"}:
                continue
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            turns.append({"role": role, "content": content})
            if role == "user":
                user_count += 1
            elif role == "assistant":
                assistant_count += 1
            if len(turns) >= self._SESSION_TITLE_CONTEXT_MESSAGES:
                break
        if user_count < self._SESSION_TITLE_MIN_USER_MESSAGES:
            return None
        if assistant_count < self._SESSION_TITLE_MIN_ASSISTANT_MESSAGES:
            return None
        if len(turns) < 2:
            return None

        total_messages = user_count + assistant_count
        last_attempt_at = int(target.get("title_last_auto_message_count", 0) or 0)
        if (
            last_attempt_at
            and total_messages - last_attempt_at < self._SESSION_TITLE_RETRY_MESSAGE_GAP
        ):
            return None

        transcript = "\n".join(f"{item['role'].upper()}: {item['content'][:280]}" for item in turns)
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Generate a concise title for a chat session. "
                    "Return only the title text, no quotes, max 8 words. "
                    "Focus on the user's goal and concrete topic."
                ),
            },
            {
                "role": "user",
                "content": "Conversation:\n" + transcript,
            },
        ]

        try:
            response = await self.provider.chat(
                messages=prompt_messages,
                tools=None,
                model=self.model,
                max_tokens=32,
                temperature=0.2,
            )
            candidate = self._normalize_generated_title(response.content or "")
        except Exception as e:
            logger.debug(f"Session title generation skipped due to provider error: {e}")
            candidate = ""

        if candidate and self._is_low_quality_generated_title(candidate):
            candidate = ""

        if not candidate:
            fallback = next(
                (
                    str(item.get("content") or "")
                    for item in turns
                    if str(item.get("role") or "") == "user"
                ),
                "",
            )
            candidate = self._normalize_generated_title(fallback)
            if candidate and self._is_low_quality_generated_title(candidate):
                candidate = ""

        if not candidate:
            self.sessions.note_auto_title_attempt(
                conversation_key,
                session_id,
                message_count=total_messages,
            )
            return None

        normalized_default = self._SESSION_DEFAULT_TITLE.lower()
        if candidate.lower() == normalized_default:
            self.sessions.note_auto_title_attempt(
                conversation_key,
                session_id,
                message_count=total_messages,
            )
            return None

        self.sessions.rename_session(
            conversation_key,
            session_id,
            candidate,
            auto_title=False,
            source="auto",
            lock_title=False,
        )
        session.metadata["title"] = candidate
        self.sessions.save(session)
        return candidate

    async def _handle_privileged_approval(
        self,
        *,
        msg: InboundMessage,
        session: Session,
        approve: bool,
    ) -> OutboundMessage:
        session_key = msg.session_key
        pending = self._approval_store.get_pending(session_key)
        if not pending:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="No pending privileged request in this chat.",
            )

        if pending.requester_id and pending.requester_id != msg.sender_id:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Only the original requester can approve or deny this privileged request.",
            )

        if not approve:
            self._approval_store.resolve(
                session_key,
                status="denied",
                resolver_id=msg.sender_id,
                result_preview="Denied by user",
            )
            session.add_message("user", msg.content)
            session.add_message("assistant", "Privileged request denied.")
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Privileged request denied.",
            )

        if not self.exec_config.privileged_enabled or not self._privileged_client:
            if os.name != "posix":
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Privileged execution is supported only on Unix/Linux.",
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "Privileged execution is not set up. "
                    "Run `nanobot privileged setup` once, then retry `/approve`."
                ),
            )

        result = await self._privileged_client.execute(
            request_id=pending.request_id,
            action=pending.action,
            action_args=pending.action_args,
            timeout_s=max(self.exec_config.timeout, 120),
        )
        ok = bool(result.get("ok"))
        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        error = str(result.get("error") or "").strip()

        parts: list[str] = []
        if ok:
            parts.append(f"Privileged request executed: {pending.action}")
        else:
            parts.append(f"Privileged request failed: {pending.action}")
        if stdout:
            parts.append("STDOUT:\n" + self._truncate_preview(stdout))
        if stderr:
            parts.append("STDERR:\n" + self._truncate_preview(stderr))
        if error:
            parts.append("Error: " + error)

        preview = self._truncate_preview("\n\n".join(parts), max_len=1200)
        self._approval_store.resolve(
            session_key,
            status="executed" if ok else "failed",
            resolver_id=msg.sender_id,
            result_preview=preview,
        )
        if not ok:
            session.add_message("user", msg.content)
            session.add_message("assistant", preview)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=preview,
            )

        # Continue the original task after successful privileged execution.
        # Keep this as an explicit system event so the model can decide next steps
        # and then end with complete_task.
        followup_event = (
            "System event: pending privileged request approved and executed.\n"
            f"Action: {pending.action}\n"
            f"Command: {pending.command}\n"
            "Execution summary:\n"
            f"{preview}\n\n"
            "Continue the original user request in this chat using these results. "
            "If the task is complete, call complete_task(final_answer=...)."
        )
        self._set_tool_context(msg.channel, msg.chat_id, msg.sender_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=followup_event,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used, llm_metadata = await self._run_agent_loop(
            initial_messages,
            initial_external_progress=True,
            request_mode=self._REQUEST_MODE_DO,
        )
        if not final_content:
            final_content = preview

        session.add_message("user", msg.content)
        session.add_message(
            "assistant",
            final_content,
            tools_used=tools_used if tools_used else None,
        )
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=self._merge_outbound_metadata(msg.metadata, llm_metadata),
        )

    @staticmethod
    def _extract_web_search_trace(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(metadata, dict):
            return []
        trace = metadata.get("web_search_trace")
        if not isinstance(trace, list):
            return []
        return [item for item in trace if isinstance(item, dict)]

    @staticmethod
    def _extract_completion_answer_from_text(text: str) -> str | None:
        """Recover final_answer when model emits completion payload as plain text JSON."""
        raw = text.strip()
        if not raw:
            return None

        candidates = [raw]
        fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw, flags=re.IGNORECASE)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

        for candidate in candidates:
            if not candidate.startswith("{"):
                continue
            try:
                parsed = json.loads(candidate)
            except Exception:
                try:
                    parsed = json_repair.loads(candidate)
                except Exception:
                    continue
            if not isinstance(parsed, dict):
                continue
            final_answer = parsed.get("final_answer")
            if isinstance(final_answer, str):
                answer = final_answer.strip()
                if answer:
                    return answer
        return None

    @staticmethod
    def _extract_completion_payload(arguments: dict[str, Any] | None) -> dict[str, Any] | None:
        """Extract completion payload fields from complete_task arguments."""
        if not isinstance(arguments, dict):
            return None

        final_answer = arguments.get("final_answer")
        if not isinstance(final_answer, str) or not final_answer.strip():
            return None

        artifacts = arguments.get("artifacts")
        evidence = arguments.get("evidence")
        actions_taken = arguments.get("actions_taken")
        return {
            "final_answer": final_answer.strip(),
            "artifacts": artifacts if isinstance(artifacts, list) else [],
            "evidence": evidence if isinstance(evidence, list) else [],
            "actions_taken": actions_taken if isinstance(actions_taken, list) else [],
        }

    @staticmethod
    def _completion_has_required_evidence(payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        evidence = payload.get("evidence")
        actions_taken = payload.get("actions_taken")
        return bool(isinstance(evidence, list) and evidence) and bool(
            isinstance(actions_taken, list) and actions_taken
        )

    @staticmethod
    def _tool_result_success(tool_name: str, result: str) -> bool:
        """Best-effort tool success detection for completion gating."""
        text = result.strip()
        if not text:
            return True

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if parsed.get("approval_required") is True:
                    return False
                if parsed.get("pending") is True:
                    return False
                status = parsed.get("status")
                if isinstance(status, str) and status.lower() in {"error", "failed"}:
                    return False
                err = parsed.get("error")
                if isinstance(err, str) and err.strip():
                    return False
                if parsed.get("ok") is False:
                    return False
        except Exception:
            pass

        lower = text.lower()
        if re.search(r"\bexit code:\s*(?!0\b)\d+", lower):
            return False
        if lower.startswith("error"):
            return False
        for marker in ("error:", "traceback", "exception", "module not found", "failed to"):
            if marker in lower:
                return False
        return True

    @classmethod
    def _compact_mode_line(cls, text: str) -> str:
        collapsed = " ".join(text.strip().split())
        if len(collapsed) > cls._MODE_CLASSIFIER_ITEM_MAX_CHARS:
            return collapsed[: cls._MODE_CLASSIFIER_ITEM_MAX_CHARS - 3] + "..."
        return collapsed

    async def _classify_request_mode(self, session: Session, user_text: str) -> tuple[str, str]:
        recent_lines: list[str] = []
        for msg in session.messages[-self._MODE_CLASSIFIER_MAX_HISTORY :]:
            role = str(msg.get("role") or "").upper()
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            recent_lines.append(f"{role}: {self._compact_mode_line(content)}")

        conversation = "\n".join(recent_lines) if recent_lines else "(none)"
        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "Classify the current turn for orchestration. "
                    "Return JSON only with keys: mode, reason. "
                    "mode must be DO or CHAT.\n"
                    "- DO: user requests concrete execution/build/fix/search/run/verification work, "
                    "or gives a follow-up control command that should continue an active task.\n"
                    "- CHAT: user asks conceptual questions, policy discussion, meta feedback, or casual conversation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Recent conversation:\n{conversation}\n\n"
                    f"Current user message:\n{user_text}\n\n"
                    "Respond with JSON only."
                ),
            },
        ]

        try:
            response = await self.provider.chat(
                messages=classifier_messages,
                tools=[],
                model=self.model,
                max_tokens=120,
                temperature=0.0,
            )
            raw = (response.content or "").strip()
            if not raw:
                raise ValueError("empty classifier response")
            if response.has_tool_calls:
                raise ValueError("classifier returned tool call")
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json_repair.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("classifier response is not a JSON object")

            mode_value = str(parsed.get("mode") or "").strip().upper()
            if mode_value not in {self._REQUEST_MODE_DO, self._REQUEST_MODE_CHAT}:
                raise ValueError(f"unknown mode: {mode_value}")

            reason = str(parsed.get("reason") or "").strip() or "classifier decision"
            return mode_value, reason
        except Exception as e:
            previous = str(session.metadata.get("last_request_mode") or "").strip().upper()
            fallback = (
                previous
                if previous in {self._REQUEST_MODE_DO, self._REQUEST_MODE_CHAT}
                else self._REQUEST_MODE_CHAT
            )
            return fallback, f"classifier fallback ({type(e).__name__})"

    def _has_task_execution_tools(self) -> bool:
        task_tools = [
            name for name in self.tools.tool_names if name not in self._NON_PROGRESS_TOOLS
        ]
        return bool(task_tools)

    @staticmethod
    def _pending_approval_message(tool_name: str, result: str) -> str | None:
        """Extract user-facing pending-approval notice from a tool result payload."""
        if tool_name != "exec":
            return None
        try:
            parsed = json.loads(result.strip())
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        if parsed.get("approval_required") is not True or parsed.get("pending") is not True:
            return None

        request_id = parsed.get("request_id")
        action = parsed.get("action")
        msg = parsed.get("message")
        lines = [
            "Privileged execution is pending user approval.",
            "Reply with /approve to continue or /deny to cancel.",
        ]
        if isinstance(request_id, str) and request_id:
            lines.insert(1, f"Request ID: {request_id}")
        if isinstance(action, str) and action:
            lines.insert(2 if len(lines) > 2 else 1, f"Action: {action}")
        if isinstance(msg, str) and msg.strip():
            lines.append(msg.strip())
        return "\n".join(lines)

    @staticmethod
    def _is_agent_browser_command(command: str) -> bool:
        return bool(re.search(r"\bagent-browser\b", command))

    @staticmethod
    def _is_agent_browser_close_command(command: str) -> bool:
        return bool(re.search(r"\bagent-browser\s+close\b", command))

    def _load_prefill_prompt(self) -> str:
        """Load optional response prefill guidance from workspace files."""
        for rel_path in self._PREFILL_FILE_CANDIDATES:
            prefill_path = self.workspace / rel_path
            if not prefill_path.exists():
                continue
            try:
                return prefill_path.read_text(encoding="utf-8").strip()
            except OSError as e:
                logger.warning(f"Failed to read prefill file {prefill_path}: {e}")
                return ""
        return ""

    @staticmethod
    def _append_prefill_tail(
        messages: list[dict[str, Any]],
        prefill_prompt: str,
    ) -> list[dict[str, Any]]:
        """Append assistant-style prefill as the final request item."""
        if not prefill_prompt:
            return messages
        return [*messages, {"role": "assistant", "content": prefill_prompt}]

    @staticmethod
    def _merge_outbound_metadata(
        base: dict[str, Any] | None, llm_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(base or {})
        if not llm_metadata:
            return merged

        nanobot_meta = merged.get("_nanobot")
        if not isinstance(nanobot_meta, dict):
            nanobot_meta = {}

        for key, value in llm_metadata.items():
            if isinstance(value, list) and isinstance(nanobot_meta.get(key), list):
                nanobot_meta[key] = [*nanobot_meta[key], *value]
            else:
                nanobot_meta[key] = value

        merged["_nanobot"] = nanobot_meta
        return merged

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        *,
        initial_external_progress: bool = False,
        request_mode: str = _REQUEST_MODE_CHAT,
    ) -> tuple[str | None, list[str], dict[str, Any]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used, llm_metadata).
        """
        messages = initial_messages
        do_mode = request_mode == self._REQUEST_MODE_DO
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        web_search_trace: list[dict[str, Any]] = []
        successful_external_actions = 1 if initial_external_progress else 0
        external_tool_attempted = False
        meaningful_tool_attempted = False
        meaningful_tool_succeeded = bool(initial_external_progress)
        no_tool_text_rounds = 0
        no_tool_empty_rounds = 0
        last_nonempty_no_tool_text = ""
        agent_browser_used = False
        agent_browser_closed = False
        prefill_prompt = self._load_prefill_prompt()

        while iteration < self.max_iterations:
            iteration += 1

            request_messages = self._append_prefill_tail(messages, prefill_prompt)
            response = await self.provider.chat(
                messages=request_messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            round_web_search_trace = self._extract_web_search_trace(response.metadata)
            web_search_trace.extend(round_web_search_trace)
            has_external_progress = successful_external_actions > 0 or bool(web_search_trace)
            if response.finish_reason == "error":
                final_content = response.content
                break

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                completion_answer: str | None = None
                completion_requested = False
                completion_payload: dict[str, Any] | None = None
                completion_schema_ok = True
                pending_approval_notice: str | None = None
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    if tool_call.name != self._COMPLETE_TOOL_NAME:
                        external_tool_attempted = True
                    if tool_call.name not in self._NON_PROGRESS_TOOLS:
                        meaningful_tool_attempted = True
                    if tool_call.name == "exec":
                        cmd_text = str(tool_call.arguments.get("command", ""))
                        if self._is_agent_browser_command(cmd_text):
                            agent_browser_used = True
                        if self._is_agent_browser_close_command(cmd_text):
                            agent_browser_closed = True
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    pending_approval_notice = self._pending_approval_message(tool_call.name, result)
                    if pending_approval_notice:
                        logger.info(f"Tool result: {tool_call.name} -> pending_approval")
                        break
                    if tool_call.name == self._COMPLETE_TOOL_NAME:
                        completion_requested = True
                        completion_payload = self._extract_completion_payload(tool_call.arguments)
                        completion_answer = (
                            completion_payload["final_answer"] if completion_payload else None
                        )
                        completion_schema_ok = self._tool_result_success(tool_call.name, result)
                        if not completion_answer:
                            logger.warning(
                                "complete_task called without final_answer; continuing loop"
                            )
                        continue

                    is_success = self._tool_result_success(tool_call.name, result)
                    status = "ok" if is_success else "failed"
                    logger.info(f"Tool result: {tool_call.name} -> {status}")
                    if is_success:
                        successful_external_actions += 1
                        if tool_call.name not in self._NON_PROGRESS_TOOLS:
                            meaningful_tool_succeeded = True

                if pending_approval_notice:
                    final_content = pending_approval_notice
                    break

                has_external_progress = successful_external_actions > 0 or bool(web_search_trace)
                no_tool_text_rounds = 0
                no_tool_empty_rounds = 0
                if completion_answer:
                    if not completion_schema_ok:
                        logger.warning("complete_task rejected: invalid payload schema")
                        messages.append(
                            {
                                "role": "user",
                                "content": self._ACTION_RETRY_REASON_INVALID_COMPLETE_PAYLOAD,
                            }
                        )
                        continue
                    if do_mode and not self._completion_has_required_evidence(completion_payload):
                        logger.warning(
                            "complete_task rejected: missing required evidence/actions in DO mode"
                        )
                        messages.append(
                            {"role": "user", "content": self._ACTION_RETRY_REASON_MISSING_EVIDENCE}
                        )
                        continue
                    if do_mode and not (meaningful_tool_succeeded or has_external_progress):
                        logger.warning(
                            "complete_task rejected: DO mode completion without verified external progress"
                        )
                        messages.append(
                            {"role": "user", "content": self._ACTION_RETRY_REASON_NO_PROGRESS}
                        )
                        continue
                    if external_tool_attempted and not meaningful_tool_attempted:
                        logger.warning(
                            "complete_task rejected: only report_to_user/non-meaningful tools observed"
                        )
                        messages.append(
                            {"role": "user", "content": self._ACTION_RETRY_REASON_REPORT_ONLY}
                        )
                        continue
                    final_content = completion_answer
                    break
                if completion_requested:
                    followup_nudge = self._COMPLETION_REJECT_NUDGE
                    if do_mode:
                        followup_nudge += (
                            " In TASK_REQUEST mode include non-empty evidence/actions_taken."
                        )
                    messages.append({"role": "user", "content": followup_nudge})
                else:
                    continue_nudge = (
                        "Reflect on the tool results and continue. "
                        "Call complete_task(final_answer=...) only when fully done."
                    )
                    if do_mode:
                        continue_nudge += (
                            " Keep executing tools and gather concrete evidence before completion."
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": continue_nudge,
                        }
                    )
            else:
                assistant_text = (response.content or "").strip()
                if assistant_text:
                    last_nonempty_no_tool_text = assistant_text
                    no_tool_text_rounds += 1
                    no_tool_empty_rounds = 0
                else:
                    no_tool_empty_rounds += 1

                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    reasoning_content=response.reasoning_content,
                )

                completion_from_text = (
                    self._extract_completion_answer_from_text(assistant_text)
                    if assistant_text
                    else None
                )
                if completion_from_text:
                    if do_mode:
                        logger.warning(
                            "Rejected text-only completion payload in DO mode; evidence-bearing complete_task required"
                        )
                        messages.append(
                            {"role": "user", "content": self._ACTION_RETRY_REASON_MISSING_EVIDENCE}
                        )
                    else:
                        logger.warning(
                            "Recovered final_answer from no-tool text payload; finalizing turn"
                        )
                        final_content = completion_from_text
                        break

                if no_tool_empty_rounds >= self._MAX_NO_TOOL_EMPTY_ROUNDS:
                    logger.warning(
                        "LLM returned empty/no-tool responses repeatedly; finalizing fallback response"
                    )
                    final_content = last_nonempty_no_tool_text or self._NO_TOOL_FALLBACK
                    break

                if no_tool_text_rounds >= self._MAX_NO_TOOL_TEXT_ROUNDS:
                    logger.warning(
                        "LLM returned no-tool text repeatedly; finalizing latest response fallback"
                    )
                    if do_mode:
                        final_content = self._NO_TOOL_FALLBACK
                    else:
                        final_content = last_nonempty_no_tool_text or self._NO_TOOL_FALLBACK
                    break

                if do_mode:
                    nudge = (
                        self._ACTION_RETRY_REASON_NO_PROGRESS
                        if external_tool_attempted and not has_external_progress
                        else self._ACTION_REQUEST_NUDGE
                    )
                else:
                    nudge = self._NO_ACTION_NUDGE
                messages.append({"role": "user", "content": nudge})

        if final_content is None:
            final_content = "I couldn't complete the task within the iteration limit."

        # Safety cleanup: if agent-browser was used but not closed in this turn,
        # close it best-effort to avoid leaked Chromium processes.
        if agent_browser_used and not agent_browser_closed and self.tools.has("exec"):
            cleanup_result = await self.tools.execute(
                "exec",
                {"command": self._AGENT_BROWSER_AUTO_CLOSE_CMD},
            )
            cleanup_ok = self._tool_result_success("exec", cleanup_result)
            if cleanup_ok:
                logger.info("Auto cleanup: agent-browser close executed")
            else:
                logger.warning(f"Auto cleanup: agent-browser close failed: {cleanup_result[:200]}")

        llm_metadata: dict[str, Any] = {}
        if web_search_trace:
            llm_metadata["web_search_trace"] = web_search_trace
        llm_metadata["request_mode"] = request_mode

        return final_content, tools_used, llm_metadata

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                try:
                    async with self._process_lock:
                        response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"Sorry, I encountered an error: {str(e)}",
                        )
                    )
            except asyncio.TimeoutError:
                continue

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self, msg: InboundMessage, session_key: str | None = None
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).

        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        conversation_key = self._conversation_key(msg.channel, msg.chat_id)
        resolved_session_id = ""
        if session_key is not None:
            session = self.sessions.get_or_create(session_key)
        else:
            requested_session_id = self._extract_requested_session_id(msg.metadata)
            session, descriptor = self.sessions.get_or_create_for_conversation(
                conversation_key,
                requested_session_id=requested_session_id,
            )
            resolved_session_id = str(descriptor.get("id") or "")

        # Handle slash commands
        cmd = msg.content.strip().lower()
        cmd_token = cmd.split()[0] if cmd else ""
        cmd_name = cmd_token.split("@", 1)[0]
        if cmd_name == "/help":
            help_text = (
                "nanobot commands:\n"
                "/new - Start and switch to a new session\n"
                "/sessions - List sessions in this chat\n"
                "/session ... - Manage sessions (list/current/new/switch/rename/delete)\n"
                "/help - Show available commands\n"
                "/approve - Approve pending privileged request\n"
                "/deny - Deny pending privileged request"
            )
            help_metadata = dict(msg.metadata)
            if session_key is None:
                snapshot = self.sessions.list_conversation_sessions(conversation_key)
                help_metadata = self._merge_outbound_metadata(
                    help_metadata,
                    self._session_state_metadata(snapshot),
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=help_text,
                metadata=help_metadata,
            )
        if cmd_name == "/approve":
            return await self._handle_privileged_approval(msg=msg, session=session, approve=True)
        if cmd_name == "/deny":
            return await self._handle_privileged_approval(msg=msg, session=session, approve=False)

        if session_cmd_response := self._handle_session_command(
            msg=msg,
            cmd_name=cmd_name,
            conversation_key=conversation_key,
        ):
            return session_cmd_response

        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        request_mode, mode_reason = await self._classify_request_mode(session, msg.content)
        logger.info(f"Request mode for {msg.channel}:{msg.chat_id}: {request_mode} ({mode_reason})")
        if request_mode == self._REQUEST_MODE_DO and not self._has_task_execution_tools():
            final_content = self._DO_MODE_NO_TOOL_RESPONSE
            session.add_message("user", msg.content)
            session.add_message("assistant", final_content)
            session.metadata["last_request_mode"] = request_mode
            session.metadata["last_request_mode_reason"] = mode_reason
            self.sessions.save(session)
            if resolved_session_id:
                await self._maybe_generate_session_title(
                    conversation_key=conversation_key,
                    session_id=resolved_session_id,
                    session=session,
                )

            outbound_metadata = dict(msg.metadata)
            if session_key is None:
                snapshot = self.sessions.list_conversation_sessions(conversation_key)
                outbound_metadata = self._merge_outbound_metadata(
                    outbound_metadata,
                    self._session_state_metadata(snapshot),
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content,
                metadata=outbound_metadata,
            )

        self._set_tool_context(msg.channel, msg.chat_id, msg.sender_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used, llm_metadata = await self._run_agent_loop(
            initial_messages,
            request_mode=request_mode,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        session.add_message("user", msg.content)
        session.add_message(
            "assistant", final_content, tools_used=tools_used if tools_used else None
        )
        session.metadata["last_request_mode"] = request_mode
        session.metadata["last_request_mode_reason"] = mode_reason
        self.sessions.save(session)

        if resolved_session_id:
            await self._maybe_generate_session_title(
                conversation_key=conversation_key,
                session_id=resolved_session_id,
                session=session,
            )

        outbound_metadata = self._merge_outbound_metadata(msg.metadata, llm_metadata)
        if session_key is None:
            snapshot = self.sessions.list_conversation_sessions(conversation_key)
            outbound_metadata = self._merge_outbound_metadata(
                outbound_metadata,
                self._session_state_metadata(snapshot),
            )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=outbound_metadata,
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        conversation_key = self._conversation_key(origin_channel, origin_chat_id)
        requested_session_id = self._extract_requested_session_id(msg.metadata)
        session, descriptor = self.sessions.get_or_create_for_conversation(
            conversation_key,
            requested_session_id=requested_session_id,
        )
        self._set_tool_context(origin_channel, origin_chat_id, msg.sender_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _, llm_metadata = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."

        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        if sid := str(descriptor.get("id") or ""):
            await self._maybe_generate_session_title(
                conversation_key=conversation_key,
                session_id=sid,
                session=session,
            )

        outbound_metadata = self._merge_outbound_metadata(msg.metadata, llm_metadata)
        snapshot = self.sessions.list_conversation_sessions(conversation_key)
        outbound_metadata = self._merge_outbound_metadata(
            outbound_metadata,
            self._session_state_metadata(snapshot),
        )

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content,
            metadata=outbound_metadata,
        )

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(
                f"Memory consolidation (archive_all): {len(session.messages)} total messages archived"
            )
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(
                    f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})"
                )
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(
                    f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})"
                )
                return

            old_messages = session.messages[session.last_consolidated : -keep_count]
            if not old_messages:
                return
            logger.info(
                f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep"
            )

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(
                f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}"
            )
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory consolidation agent. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(
                    f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}"
                )
                return

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(
                f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}"
            )
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).

        Returns:
            The agent's response.
        """
        response = await self.process_direct_message(
            content=content,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
        )
        return response.content if response else ""

    async def process_direct_message(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> OutboundMessage | None:
        """Process a message directly and return the full outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        async with self._process_lock:
            return await self._process_message(msg, session_key=session_key)
