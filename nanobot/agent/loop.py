"""Agent loop: the core processing engine."""

import asyncio
from contextlib import AsyncExitStack, nullcontext
import inspect
import json
import json_repair
import os
from pathlib import Path
import re
import time
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.complete import CompleteTaskTool
from nanobot.agent.tools.memory import MemoryTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.report import ReportToUserTool
from nanobot.agent.tools.sessions import SessionsTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.memory import MemoryConsolidator, MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.security.approval_store import ApprovalStore
from nanobot.security.privileged_client import PrivilegedClient
from nanobot.session.manager import Session, SessionManager
from nanobot.observability.langsmith import get_langsmith_tracer
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
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
    _REQUEST_INTENT_TASK = "TASK"
    _REQUEST_INTENT_CONTROL = "CONTROL"
    _REQUEST_INTENT_META = "META"
    _REQUEST_INTENT_CASUAL = "CASUAL"
    _REQUEST_EXEC_REQUIRED = "REQUIRED"
    _REQUEST_EXEC_OPTIONAL = "OPTIONAL"
    _REQUEST_EXEC_FORBIDDEN = "FORBIDDEN"
    _NON_PROGRESS_TOOLS = {_COMPLETE_TOOL_NAME, _REPORT_TOOL_NAME}
    _MODE_CLASSIFIER_MAX_HISTORY = 10
    _MODE_CLASSIFIER_ITEM_MAX_CHARS = 220
    _MODE_CLASSIFIER_MAX_CAPABILITY_ITEMS = 30
    _REQUIRED_EXEC_NO_TOOL_RESPONSE = (
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
        "Retry reason: complete_task in execution=REQUIRED mode requires non-empty `evidence` and `actions_taken`. "
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
    _SESSION_TRACE_MESSAGES_KEY = "_session_trace_messages"
    _SKIP_SESSION_ASSISTANT_KEY = "_skip_session_assistant"
    _SESSION_TRACE_MAX_EVENTS = 40
    _SESSION_TRACE_RESULT_MAX_CHARS = 1200
    _PROGRESS_HINT_HIDE_TOOLS = {"exec"}
    _TRACE_RECENT_MESSAGES_LIMIT = 8
    _TRACE_MESSAGE_PREVIEW_MAX_CHARS = 500
    _NO_TOOL_FALLBACK = (
        "I couldn't make progress with tool execution or completion signaling. "
        "Please provide a more specific next instruction."
    )
    _AGENT_BROWSER_AUTO_CLOSE_CMD = "agent-browser close >/dev/null 2>&1 || true"
    _APP_SERVER_PROGRESS_MIN_CHARS = 24
    _APP_SERVER_PROGRESS_FORCE_FLUSH_CHARS = 160

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 30,
        context_window_tokens: int = 65_536,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        memory_window: int = 50,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        web_search_config: Any | None = None,
        web_proxy: str | None = None,
        channels_config: Any | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig

        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.channels_config = channels_config
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_search_config = web_search_config
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        self.context = ContextBuilder(workspace)
        self.memory = self.context.memory
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._background_tasks: list[asyncio.Task[Any]] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        max_concurrent = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "0") or "0")
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )
        try:
            approvals_path = get_data_path() / "approvals" / "requests.json"
            self._approval_store = ApprovalStore(
                approvals_path,
                ttl_seconds=self.exec_config.approval_ttl_sec,
                single_pending_per_chat=self.exec_config.single_pending_per_chat,
            )
        except OSError:
            approvals_path = Path.home() / ".nanobot" / "approvals" / "requests.json"
            fallback_path = self.workspace / "approvals" / "requests.json"
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
        provider_generation = getattr(provider, "generation", None)
        provider_max_completion = getattr(provider_generation, "max_tokens", None)
        if not isinstance(provider_max_completion, int) or provider_max_completion < 0:
            provider_max_completion = self.max_tokens
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider_max_completion,
        )
        self._process_lock = asyncio.Lock()
        self._active_tasks: dict[str, list[asyncio.Task[Any]]] = {}
        self._consolidation_lock = asyncio.Lock()
        self._consolidation_tasks: dict[str, asyncio.Task[bool]] = {}
        self._consolidation_scheduled_counts: dict[str, int] = {}
        self._tracer = get_langsmith_tracer()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    deny_patterns=self.exec_config.deny_patterns,
                    restrict_to_workspace=self.restrict_to_workspace,
                    path_append=self.exec_config.path_append,
                    privileged_enabled=self.exec_config.privileged_enabled,
                    approval_store=self._approval_store,
                )
            )

        # Web tools
        native_web_search = getattr(self.provider, "supports_native_web_search", False)
        if not isinstance(native_web_search, bool):
            native_web_search = False
        if not native_web_search:
            if self.web_search_config is not None:
                self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
            else:
                self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(CompleteTaskTool())
        self.tools.register(SessionsTool(self.sessions))
        self.tools.register(MemoryTool(self.memory))

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
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
            self._mcp_connected = False
        finally:
            self._mcp_connecting = False

    def _set_tool_context(
        self,
        channel: str,
        chat_id: str,
        sender_id: str = "",
        message_id: str | None = None,
        *,
        lookup_session_key: str | None = None,
        session: Session | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        if exec_tool := self.tools.get("exec"):
            if isinstance(exec_tool, ExecTool):
                exec_tool.set_context(
                    channel,
                    chat_id,
                    sender_id,
                    lookup_session_key=lookup_session_key,
                    current_session_id=session.id if session else None,
                    current_session_key=session.key if session else None,
                    origin_session_id=session.id if session else None,
                    origin_session_key=session.key if session else None,
                )

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

        if report_tool := self.tools.get("report_to_user"):
            if isinstance(report_tool, ReportToUserTool):
                report_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(
                    channel,
                    chat_id,
                    session_key=session.key if session else lookup_session_key,
                    session_id=session.id if session else None,
                )

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _truncate_preview(text: str, max_len: int = 800) -> str:
        clean = text.strip()
        if len(clean) <= max_len:
            return clean
        return clean[:max_len] + f"\n... (truncated, {len(clean) - max_len} more chars)"

    async def _provider_chat(self, **kwargs: Any):
        """Call the provider with compatibility for old/new provider test doubles."""
        for method_name in ("chat_with_retry", "chat"):
            method = getattr(self.provider, method_name, None)
            if not callable(method) or not inspect.iscoroutinefunction(method):
                continue
            try:
                result = method(**kwargs)
                if inspect.isawaitable(result):
                    return await result
                if isinstance(result, LLMResponse):
                    return result
            except TypeError as exc:
                if "reasoning_effort" not in str(exc):
                    raise
                compat_kwargs = dict(kwargs)
                compat_kwargs.pop("reasoning_effort", None)
                result = method(**compat_kwargs)
                if inspect.isawaitable(result):
                    return await result
                if isinstance(result, LLMResponse):
                    return result
        for method_name in ("chat_with_retry", "chat"):
            method = getattr(self.provider, method_name, None)
            if not callable(method):
                continue
            result = method(**kwargs)
            if inspect.isawaitable(result):
                return await result
            if isinstance(result, LLMResponse):
                return result
        raise AttributeError("Provider does not expose chat/chat_with_retry")

    def _uses_app_server_runtime(self) -> bool:
        """Whether primary turns should run through Codex App Server."""
        uses_app_server = getattr(self.provider, "uses_app_server", False)
        return uses_app_server if isinstance(uses_app_server, bool) else False

    async def _run_app_server_primary_turn(
        self,
        *,
        session: Session,
        msg: InboundMessage,
        request_execution: str,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], dict[str, Any]]:
        """Execute the current turn via Codex App Server."""
        existing_thread_id = str(session.metadata.get("app_server_thread_id") or "").strip() or None
        history = session.get_history(max_messages=self.memory_window)
        working_set_path = None
        working_set_text = ""
        if existing_thread_id is None and (
            session.messages or session.summary.strip() or session.title.strip()
        ):
            working_set_path, working_set_text = self.sessions.artifacts.load_working_set(session)
        input_items = self.context.build_app_server_turn_input(
            current_message=msg.content,
            history=history,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            bootstrap_history=existing_thread_id is None and bool(history),
            working_set_text=working_set_text,
            working_set_path=str(working_set_path) if working_set_path else None,
        )
        progress_buffer = ""
        progress_phase = ""

        def _should_flush_progress(text: str, *, force: bool) -> bool:
            if not text:
                return False
            if len(text) >= self._APP_SERVER_PROGRESS_FORCE_FLUSH_CHARS:
                return True
            if re.search(r"(?:[.!?…。:]\s*|\n{2,})$", text):
                return True
            if force and len(text) >= self._APP_SERVER_PROGRESS_MIN_CHARS:
                return True
            return False

        async def _flush_progress(*, force: bool) -> None:
            nonlocal progress_buffer, progress_phase
            if not on_progress:
                progress_buffer = ""
                progress_phase = ""
                return

            text = progress_buffer.strip()
            if not _should_flush_progress(text, force=force):
                if force:
                    progress_buffer = ""
                    progress_phase = ""
                return

            progress_buffer = ""
            progress_phase = ""
            await on_progress(text)

        async def _app_server_event(event: dict[str, Any]) -> None:
            nonlocal progress_buffer, progress_phase
            event_type = str(event.get("type") or "")
            if event_type == "tool_call":
                await _flush_progress(force=True)
                tool_name = str(event.get("tool") or "").strip()
                arguments = event.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                tool_hint = self._app_server_tool_hint(tool_name, arguments)
                if tool_hint and on_progress:
                    await on_progress("", tool_hint=tool_hint)
                return

            if event_type == "tool_result":
                tool_name = str(event.get("tool") or "").strip()
                logger.info(
                    "Codex App Server tool result: {} -> {}",
                    tool_name or "(unknown)",
                    "ok" if event.get("success") else "failed",
                )
                return

            if event_type == "agent_delta":
                phase = str(event.get("phase") or "").strip().lower()
                if phase == "final_answer":
                    return
                delta = self._remove_think_tags(str(event.get("delta") or ""))
                if not delta:
                    return
                if progress_phase and phase and phase != progress_phase:
                    await _flush_progress(force=True)
                if phase:
                    progress_phase = phase
                progress_buffer += delta
                await _flush_progress(force=False)
                return

        result = await self.provider.run_app_server_turn(
            thread_id=existing_thread_id,
            input_items=input_items,
            tools=self.tools,
            developer_instructions=self.context.build_app_server_prompt(),
            event_callback=_app_server_event,
            cwd=str(self.workspace),
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            exclude_tool_names=[self._COMPLETE_TOOL_NAME],
        )
        await _flush_progress(force=True)
        session.metadata["app_server_thread_id"] = result.thread_id
        session.metadata["app_server_last_turn_id"] = result.turn_id

        llm_metadata = dict(result.metadata or {})
        llm_metadata["request_execution"] = request_execution
        llm_metadata["app_server_thread_id"] = result.thread_id
        llm_metadata["app_server_turn_id"] = result.turn_id
        return result.final_text, list(result.tools_used), llm_metadata

    @staticmethod
    def _task_session_key(session: Session) -> str:
        return session.conversation_key or session.key

    def _resolve_session(
        self,
        msg: InboundMessage,
        *,
        session_key: str | None = None,
    ) -> tuple[Session, str, bool]:
        fixed_session_mode = session_key is not None
        if fixed_session_mode:
            session = self.sessions.get_or_create(session_key)
            return session, session.key, True

        conversation_key = msg.session_key
        session = self.sessions.get_active_session(conversation_key)
        return session, conversation_key, False

    def _track_consolidation_task(self, session: Session) -> asyncio.Future[bool]:
        existing = self._consolidation_tasks.get(session.key)
        if existing and not existing.done():
            return existing
        scheduled_count = self._consolidation_scheduled_counts.get(session.key)
        if scheduled_count is not None:
            baseline = max(int(session.last_consolidated), scheduled_count)
            if len(session.messages) <= baseline + 2:
                loop = asyncio.get_running_loop()
                done: asyncio.Future[bool] = loop.create_future()
                done.set_result(True)
                return done

        task = asyncio.create_task(self._run_serialized_consolidation(session))
        self._consolidation_tasks[session.key] = task
        self._consolidation_scheduled_counts[session.key] = len(session.messages)

        def _cleanup(done: asyncio.Task[bool]) -> None:
            current = self._consolidation_tasks.get(session.key)
            if current is done:
                self._consolidation_tasks.pop(session.key, None)

        task.add_done_callback(_cleanup)
        return task

    async def _run_serialized_consolidation(
        self,
        session: Session,
        *,
        archive_all: bool = False,
    ) -> bool:
        async with self._consolidation_lock:
            result = await self._consolidate_memory(session, archive_all=archive_all)
        if result is not False and getattr(session, "id", None):
            self.sessions.save(session)
        return result is not False

    def _refresh_session(self, session: Session) -> Session:
        if session.id and (refreshed := self.sessions.get_by_id(session.id)):
            return refreshed
        return self.sessions.get_or_create(session.key)

    async def _await_inflight_consolidations(self) -> None:
        tasks = [task for task in self._consolidation_tasks.values() if not task.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)

    def _get_approval_target_session(self, pending: Any, fallback_session: Session) -> Session:
        if getattr(pending, "origin_session_id", None):
            if session := self.sessions.get_by_id(pending.origin_session_id):
                return session
        if getattr(pending, "origin_session_key", None):
            return self.sessions.get_or_create(str(pending.origin_session_key))
        if getattr(pending, "current_session_id", None):
            if session := self.sessions.get_by_id(pending.current_session_id):
                return session
        if getattr(pending, "current_session_key", None):
            return self.sessions.get_or_create(str(pending.current_session_key))
        return fallback_session

    async def _handle_privileged_approval(
        self,
        *,
        msg: InboundMessage,
        session: Session,
        approval_key: str,
        approve: bool,
    ) -> OutboundMessage:
        pending = self._approval_store.get_pending(approval_key)
        if not pending:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="No pending privileged request in this chat.",
            )
        target_session = self._get_approval_target_session(pending, session)

        if pending.requester_id and pending.requester_id != msg.sender_id:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Only the original requester can approve or deny this privileged request.",
            )

        if not approve:
            self._approval_store.resolve(
                approval_key,
                status="denied",
                resolver_id=msg.sender_id,
                result_preview="Denied by user",
            )
            target_session.add_message("user", msg.content)
            target_session.add_message("assistant", "Privileged request denied.")
            self.sessions.save(target_session)
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
            approval_key,
            status="executed" if ok else "failed",
            resolver_id=msg.sender_id,
            result_preview=preview,
        )
        if not ok:
            target_session.add_message("user", msg.content)
            target_session.add_message("assistant", preview)
            self.sessions.save(target_session)
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
        self._set_tool_context(
            msg.channel,
            msg.chat_id,
            msg.sender_id,
            msg.metadata.get("message_id"),
            lookup_session_key=approval_key,
            session=target_session,
        )
        initial_messages = self.context.build_messages(
            history=target_session.get_history(max_messages=self.memory_window),
            current_message=followup_event,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used, llm_metadata = await self._run_agent_loop(
            initial_messages,
            initial_external_progress=True,
            request_execution=self._REQUEST_EXEC_REQUIRED,
        )
        session_trace_messages = llm_metadata.pop(self._SESSION_TRACE_MESSAGES_KEY, None)
        skip_session_assistant = bool(llm_metadata.pop(self._SKIP_SESSION_ASSISTANT_KEY, False))
        if not final_content:
            final_content = preview

        target_session.add_message("user", msg.content)
        self._append_session_trace_messages(target_session, session_trace_messages)
        if not skip_session_assistant:
            target_session.add_message(
                "assistant",
                final_content,
                tools_used=tools_used if tools_used else None,
            )
        self.sessions.save(target_session)

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

    @staticmethod
    def _looks_like_explicit_action_request(text: str) -> bool:
        lowered = text.lower()
        patterns = [
            r"\b(play|run|execute|open|send|search|fetch|read|write|edit|fix|build|implement|install|deploy|restart|schedule|verify)\b",
            r"(노래|음악).*(틀어|재생|play)",
            r"(실행|돌려|켜|열어|보내|찾아|가져와|수정|고쳐|설치|배포|재시작|등록|확인)(줘|해|해줘)?",
        ]
        return any(re.search(p, lowered) for p in patterns)

    async def _classify_request(self, session: Session, user_text: str) -> tuple[str, str, str]:
        recent_lines: list[str] = []
        for msg in session.messages[-self._MODE_CLASSIFIER_MAX_HISTORY :]:
            role = str(msg.get("role") or "").upper()
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            recent_lines.append(f"{role}: {self._compact_mode_line(content)}")

        conversation = "\n".join(recent_lines) if recent_lines else "(none)"
        previous_intent = str(session.metadata.get("last_request_intent") or "").strip().upper()
        previous_execution = str(session.metadata.get("last_request_execution") or "").strip().upper()
        previous_routing = (
            f"intent={previous_intent or 'N/A'}, "
            f"execution={previous_execution or 'N/A'}"
        )
        available_tools = sorted(self.tools.tool_names)
        if len(available_tools) > self._MODE_CLASSIFIER_MAX_CAPABILITY_ITEMS:
            omitted = len(available_tools) - self._MODE_CLASSIFIER_MAX_CAPABILITY_ITEMS
            available_tools = [
                *available_tools[: self._MODE_CLASSIFIER_MAX_CAPABILITY_ITEMS],
                f"... ({omitted} more tools)",
            ]
        tools_block = "\n".join(f"- {name}" for name in available_tools) if available_tools else "- (none)"

        available_skills: list[str] = []
        try:
            skill_rows = self.context.skills.list_skills(filter_unavailable=True)
            available_skills = sorted(
                {str(row.get("name")).strip() for row in skill_rows if str(row.get("name", "")).strip()}
            )
        except Exception:
            available_skills = []
        if len(available_skills) > self._MODE_CLASSIFIER_MAX_CAPABILITY_ITEMS:
            omitted = len(available_skills) - self._MODE_CLASSIFIER_MAX_CAPABILITY_ITEMS
            available_skills = [
                *available_skills[: self._MODE_CLASSIFIER_MAX_CAPABILITY_ITEMS],
                f"... ({omitted} more skills)",
            ]
        skills_block = "\n".join(f"- {name}" for name in available_skills) if available_skills else "- (none)"

        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "Classify the current turn for orchestration using two axes. "
                    "Return JSON only with keys: intent, execution, reason.\n"
                    "intent must be one of TASK, CONTROL, META, CASUAL.\n"
                    "execution must be one of REQUIRED, OPTIONAL, FORBIDDEN.\n"
                    "- REQUIRED: faithful completion requires one or more real tool actions.\n"
                    "- OPTIONAL: answer can be completed without tools; tools may be used only if needed.\n"
                    "- FORBIDDEN: do not run tools; reply directly.\n"
                    "- Decide execution by task nature, not by confidence in success.\n"
                    "- If the user requests a concrete external action (e.g., play music, send message, execute command, fetch live data), choose REQUIRED even when capability is unclear.\n"
                    "- If the user gives a control command while an active task is ongoing, set intent=CONTROL and execution=REQUIRED."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Recent conversation:\n{conversation}\n\n"
                    f"Previous routing:\n{previous_routing}\n\n"
                    f"Available tools:\n{tools_block}\n\n"
                    f"Available skills:\n{skills_block}\n\n"
                    f"Current user message:\n{user_text}\n\n"
                    "Respond with JSON only."
                ),
            },
        ]

        try:
            response = await self._provider_chat(
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

            intent_value = str(parsed.get("intent") or "").strip().upper()
            if intent_value not in {
                self._REQUEST_INTENT_TASK,
                self._REQUEST_INTENT_CONTROL,
                self._REQUEST_INTENT_META,
                self._REQUEST_INTENT_CASUAL,
            }:
                raise ValueError(f"unknown intent: {intent_value}")

            execution_value = str(parsed.get("execution") or "").strip().upper()
            if execution_value not in {
                self._REQUEST_EXEC_REQUIRED,
                self._REQUEST_EXEC_OPTIONAL,
                self._REQUEST_EXEC_FORBIDDEN,
            }:
                raise ValueError(f"unknown execution: {execution_value}")

            reason = str(parsed.get("reason") or "").strip() or "classifier decision"
            if (
                execution_value != self._REQUEST_EXEC_REQUIRED
                and self._looks_like_explicit_action_request(user_text)
            ):
                execution_value = self._REQUEST_EXEC_REQUIRED
                if intent_value in {self._REQUEST_INTENT_META, self._REQUEST_INTENT_CASUAL}:
                    intent_value = self._REQUEST_INTENT_TASK
                reason = f"{reason}; heuristic override: explicit action request"
            return intent_value, execution_value, reason
        except Exception as e:
            fallback_intent = (
                previous_intent
                if previous_intent in {
                    self._REQUEST_INTENT_TASK,
                    self._REQUEST_INTENT_CONTROL,
                    self._REQUEST_INTENT_META,
                    self._REQUEST_INTENT_CASUAL,
                }
                else self._REQUEST_INTENT_META
            )
            fallback_execution = (
                previous_execution
                if previous_execution in {
                    self._REQUEST_EXEC_REQUIRED,
                    self._REQUEST_EXEC_OPTIONAL,
                    self._REQUEST_EXEC_FORBIDDEN,
                }
                else (
                    self._REQUEST_EXEC_REQUIRED
                    if self._looks_like_explicit_action_request(user_text)
                    else self._REQUEST_EXEC_OPTIONAL
                )
            )
            if (
                fallback_execution == self._REQUEST_EXEC_REQUIRED
                and fallback_intent in {self._REQUEST_INTENT_META, self._REQUEST_INTENT_CASUAL}
            ):
                fallback_intent = self._REQUEST_INTENT_TASK
            return fallback_intent, fallback_execution, f"classifier fallback ({type(e).__name__})"

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

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _remove_think_tags(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"<think>[\s\S]*?</think>", "", text)

    def _append_session_trace_messages(
        self,
        session: Session,
        trace_messages: list[dict[str, Any]] | None,
    ) -> None:
        """Persist structured assistant/tool trace messages into session history."""
        if not isinstance(trace_messages, list) or not trace_messages:
            return

        for item in trace_messages:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            if role == "assistant":
                content = item.get("content")
                text = content if isinstance(content, str) else ""
                tool_calls = item.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    session.add_message("assistant", text, tool_calls=tool_calls)
                elif text:
                    session.add_message("assistant", text)
                continue

            if role != "tool":
                continue

            tool_call_id = item.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            content = item.get("content")
            text = content if isinstance(content, str) else str(content or "")
            kwargs: dict[str, Any] = {"tool_call_id": tool_call_id}
            tool_name = item.get("name")
            if isinstance(tool_name, str) and tool_name:
                kwargs["name"] = tool_name
            session.add_message("tool", text, **kwargs)

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        names: list[str] = []
        for tc in tool_calls:
            name = getattr(tc, "name", None)
            if not isinstance(name, str) or not name:
                continue
            if name not in names:
                names.append(name)
        if not names:
            return ""
        return f"Using tool: {names[0]}"

    @staticmethod
    def _app_server_tool_hint(tool_name: str, arguments: dict[str, Any]) -> str:
        if not tool_name:
            return ""
        return f"Using tool: {tool_name}"

    @staticmethod
    def _trace_content_text(content: Any) -> str:
        """Normalize structured content into text for trace readability."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts)
        return str(content or "")

    @staticmethod
    def _strip_request_routing_context(text: str) -> str:
        """Remove injected request-routing context blocks from trace payloads."""
        stripped = re.sub(
            r"\[(?:REQUEST_ROUTING_CONTEXT|REQUEST_MODE_CONTEXT)\][\s\S]*?\[/(?:REQUEST_ROUTING_CONTEXT|REQUEST_MODE_CONTEXT)\]\s*",
            "",
            text or "",
        ).strip()
        return stripped or (text or "").strip()

    @classmethod
    def _extract_latest_user_message(cls, messages: list[dict[str, Any]]) -> str | None:
        """Return latest user-authored text, without internal mode context wrappers."""
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            text = cls._trace_content_text(message.get("content"))
            text = cls._strip_request_routing_context(text)
            if not text:
                continue
            return cls._truncate_preview(text, max_len=cls._TRACE_MESSAGE_PREVIEW_MAX_CHARS)
        return None

    @classmethod
    def _trace_message_view(cls, message: dict[str, Any]) -> dict[str, Any]:
        role_value = message.get("role")
        role = role_value if isinstance(role_value, str) and role_value else "unknown"
        text = cls._trace_content_text(message.get("content"))
        if role == "user":
            text = cls._strip_request_routing_context(text)
        if text:
            text = cls._truncate_preview(text, max_len=cls._TRACE_MESSAGE_PREVIEW_MAX_CHARS)

        view: dict[str, Any] = {"role": role, "content": text}
        tool_name = message.get("name")
        if role == "tool" and isinstance(tool_name, str) and tool_name:
            view["name"] = tool_name
        return view

    @classmethod
    def _build_trace_focus_input(
        cls,
        messages: list[dict[str, Any]],
        *,
        request_user_message: str | None = None,
        recent_limit: int | None = None,
    ) -> dict[str, Any]:
        """Build a trace payload focused on the latest real user request."""
        limit = max(1, recent_limit or cls._TRACE_RECENT_MESSAGES_LIMIT)
        recent_raw = messages[-limit:] if messages else []
        recent_messages = [
            cls._trace_message_view(message)
            for message in recent_raw
            if isinstance(message, dict)
        ]
        payload: dict[str, Any] = {
            "latest_user_message": cls._extract_latest_user_message(messages),
            "recent_messages": recent_messages,
            "total_messages": len(messages),
            "omitted_messages": max(0, len(messages) - len(recent_messages)),
        }
        if request_user_message:
            payload["request_user_message"] = request_user_message
        return payload

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        *,
        initial_external_progress: bool = False,
        request_execution: str = _REQUEST_EXEC_OPTIONAL,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], dict[str, Any]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.
            on_progress: Optional callback for intermediate progress updates.

        Returns:
            Tuple of (final_content, list_of_tools_used, llm_metadata).
        """
        messages = initial_messages
        request_user_message = self._extract_latest_user_message(initial_messages)
        simple_completion_mode = request_user_message is None and not initial_messages
        requires_execution = request_execution == self._REQUEST_EXEC_REQUIRED
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        web_search_trace: list[dict[str, Any]] = []
        session_trace_messages: list[dict[str, Any]] = []
        trace_messages_omitted = 0
        successful_external_actions = 1 if initial_external_progress else 0
        external_tool_attempted = False
        meaningful_tool_attempted = False
        meaningful_tool_succeeded = bool(initial_external_progress)
        no_tool_text_rounds = 0
        no_tool_empty_rounds = 0
        last_nonempty_no_tool_text = ""
        last_progress_text = ""
        last_tool_hint = ""
        agent_browser_used = False
        agent_browser_closed = False
        prefill_prompt = self._load_prefill_prompt()
        llm_error = False
        flow_span = self._tracer.start_span(
            "nanobot.agent_loop",
            run_type="chain",
            inputs=self._build_trace_focus_input(
                messages,
                request_user_message=request_user_message,
            ),
            metadata={
                "request_execution": request_execution,
                "model": self.model,
                "provider": type(self.provider).__name__,
                "max_iterations": self.max_iterations,
            },
        )

        async def _emit_progress(content: str = "", *, tool_hint: str | None = None) -> None:
            if not on_progress:
                return
            try:
                await on_progress(content, tool_hint=tool_hint)
                return
            except TypeError:
                pass

            # Backward compatibility for callbacks with old signature(content).
            text = (content or "").strip()
            if not text and tool_hint:
                text = tool_hint.strip()
            if not text:
                return
            await on_progress(text)

        while iteration < self.max_iterations:
            iteration += 1

            request_messages = self._append_prefill_tail(messages, prefill_prompt)
            llm_span = self._tracer.start_span(
                "llm.chat",
                run_type="llm",
                inputs=self._build_trace_focus_input(
                    request_messages,
                    request_user_message=request_user_message,
                ),
                metadata={"iteration": iteration, "model": self.model},
            )
            try:
                response = await self._provider_chat(
                    messages=request_messages,
                    tools=self.tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
            except Exception as e:
                llm_span.finish(error=e)
                raise
            usage = response.usage or {}
            self._last_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
            llm_span.set_outputs(
                {
                    "content": response.content,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                    "reasoning_content": response.reasoning_content,
                    "thinking_blocks": response.thinking_blocks,
                    "metadata": response.metadata,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                }
            )
            llm_span.finish()
            round_web_search_trace = self._extract_web_search_trace(response.metadata)
            web_search_trace.extend(round_web_search_trace)
            has_external_progress = successful_external_actions > 0 or bool(web_search_trace)
            if response.finish_reason == "error":
                final_content = self._strip_think(response.content) or response.content
                llm_error = True
                break

            if response.has_tool_calls:
                if on_progress:
                    progress_tool_calls = [
                        tc for tc in response.tool_calls if tc.name not in self._NON_PROGRESS_TOOLS
                    ]
                    if progress_tool_calls:
                        progress_text = self._strip_think(response.content)
                        visible_hint_calls = [
                            tc
                            for tc in progress_tool_calls
                            if tc.name not in self._PROGRESS_HINT_HIDE_TOOLS
                        ]
                        tool_hint = self._tool_hint(visible_hint_calls)
                        try:
                            if progress_text:
                                normalized = " ".join(progress_text.split())
                                if normalized != last_progress_text:
                                    last_progress_text = normalized
                                    await _emit_progress(progress_text)
                            elif tool_hint:
                                normalized_hint = " ".join(tool_hint.split())
                                if normalized_hint != last_tool_hint:
                                    last_tool_hint = normalized_hint
                                    await _emit_progress(tool_hint=tool_hint)
                        except Exception as e:
                            logger.debug(f"Progress callback failed: {e}")

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
                    thinking_blocks=response.thinking_blocks,
                )
                trace_tool_call_dicts = [
                    tc for tc in tool_call_dicts
                    if ((tc.get("function") or {}).get("name") != self._COMPLETE_TOOL_NAME)
                ]
                if trace_tool_call_dicts:
                    if len(session_trace_messages) < self._SESSION_TRACE_MAX_EVENTS:
                        trace_content = self._strip_think(response.content) or ""
                        session_trace_messages.append(
                            {
                                "role": "assistant",
                                "content": trace_content,
                                "tool_calls": trace_tool_call_dicts,
                            }
                        )
                    else:
                        trace_messages_omitted += 1

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
                    tool_span = self._tracer.start_span(
                        f"tool.{tool_call.name}",
                        run_type="tool",
                        inputs=tool_call.arguments,
                        metadata={"iteration": iteration, "tool_call_id": tool_call.id},
                    )
                    try:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    except Exception as e:
                        tool_span.finish(error=e)
                        raise
                    tool_span.set_outputs({"result": result})
                    tool_span.finish()
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    if tool_call.name != self._COMPLETE_TOOL_NAME:
                        result_clean = self._strip_think(result) or result
                        result_preview = self._truncate_preview(
                            str(result_clean),
                            max_len=self._SESSION_TRACE_RESULT_MAX_CHARS,
                        )
                        if len(session_trace_messages) < self._SESSION_TRACE_MAX_EVENTS:
                            session_trace_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_call.name,
                                    "content": result_preview,
                                }
                            )
                        else:
                            trace_messages_omitted += 1

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
                    if requires_execution and not self._completion_has_required_evidence(completion_payload):
                        logger.warning(
                            "complete_task rejected: missing required evidence/actions in execution=REQUIRED"
                        )
                        messages.append(
                            {"role": "user", "content": self._ACTION_RETRY_REASON_MISSING_EVIDENCE}
                        )
                        continue
                    if requires_execution and not (meaningful_tool_succeeded or has_external_progress):
                        logger.warning(
                            "complete_task rejected: execution=REQUIRED completion without verified external progress"
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
                    if requires_execution:
                        followup_nudge += (
                            " In execution=REQUIRED mode include non-empty evidence/actions_taken."
                        )
                    messages.append({"role": "user", "content": followup_nudge})
                else:
                    continue_nudge = (
                        "Reflect on the tool results and continue. "
                        "Call complete_task(final_answer=...) only when fully done."
                    )
                    if requires_execution:
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
                assistant_text = self._strip_think(response.content) or ""
                if simple_completion_mode:
                    final_content = assistant_text or response.content
                    break
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
                    thinking_blocks=response.thinking_blocks,
                )

                completion_from_text = (
                    self._extract_completion_answer_from_text(assistant_text)
                    if assistant_text
                    else None
                )
                if completion_from_text:
                    if requires_execution:
                        logger.warning(
                            "Rejected text-only completion payload in execution=REQUIRED; evidence-bearing complete_task required"
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
                    final_content = last_nonempty_no_tool_text or self._NO_TOOL_FALLBACK
                    break

                if requires_execution:
                    no_tool_round = no_tool_text_rounds + no_tool_empty_rounds
                    reason = (
                        "Retry reason: the previous assistant response had no tool calls "
                        f"(no-tool round {no_tool_round})."
                    )
                    if assistant_text:
                        reason += (
                            " Last response summary: "
                            f"{self._truncate_preview(assistant_text, max_len=220)}"
                        )
                    nudge = (
                        self._ACTION_RETRY_REASON_NO_PROGRESS
                        if external_tool_attempted and not has_external_progress
                        else (
                            f"{reason} "
                            "This turn is execution=REQUIRED. Execute at least one relevant tool now. "
                            "If blocked, call complete_task(final_answer=...) with a concise failure reason "
                            "and the exact user action required."
                        )
                    )
                else:
                    no_tool_round = no_tool_text_rounds + no_tool_empty_rounds
                    reason = (
                        "Retry reason: the previous assistant response had no tool calls "
                        f"(no-tool round {no_tool_round})."
                    )
                    nudge = f"{reason} {self._NO_ACTION_NUDGE}"
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
        if session_trace_messages:
            if trace_messages_omitted > 0 and len(session_trace_messages) < self._SESSION_TRACE_MAX_EVENTS:
                session_trace_messages.append(
                    {
                        "role": "assistant",
                        "content": f"(tool trace truncated: {trace_messages_omitted} events omitted)",
                    }
                )
            llm_metadata[self._SESSION_TRACE_MESSAGES_KEY] = session_trace_messages
        if llm_error:
            llm_metadata[self._SKIP_SESSION_ASSISTANT_KEY] = True
        llm_metadata["request_execution"] = request_execution

        flow_span.set_outputs(
            {
                "final_content": final_content,
                "tools_used": tools_used,
                "metadata": llm_metadata,
            }
        )
        flow_span.finish()

        return final_content, tools_used, llm_metadata

    async def run(self) -> None:
        """Run the agent loop, keeping priority slash commands responsive."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue

            asyncio.create_task(self._dispatch(msg))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            stack = self._mcp_stack
            self._mcp_stack = None
            stack_close_task = asyncio.create_task(stack.aclose())
            try:
                await asyncio.shield(stack_close_task)
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            except asyncio.CancelledError:
                try:
                    await stack_close_task
                except (RuntimeError, BaseExceptionGroup):
                    pass
        self._mcp_connected = False
        self._mcp_connecting = False
        provider_close = getattr(self.provider, "aclose", None)
        if callable(provider_close):
            try:
                result = provider_close()
                if inspect.isawaitable(result):
                    close_task = asyncio.create_task(result)
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError:
                        try:
                            await close_task
                        except Exception as e:
                            logger.debug("Provider shutdown failed after cancellation: {}", e)
                    except Exception as e:
                        logger.debug("Provider shutdown failed: {}", e)
            except Exception as e:
                logger.debug("Provider shutdown failed: {}", e)

    def _schedule_background(self, coro: Awaitable[Any]) -> None:
        """Track background tasks so shutdown can drain them cleanly."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)

        def _cleanup(done: asyncio.Task[Any]) -> None:
            if done in self._background_tasks:
                self._background_tasks.remove(done)

        task.add_done_callback(_cleanup)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
            on_progress: Optional callback for intermediate progress output.

        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        session, lookup_session_key, fixed_session_mode = self._resolve_session(
            msg,
            session_key=session_key,
        )
        conversation_key = msg.session_key

        raw = msg.content.strip()
        command_ctx = CommandContext(
            msg=msg,
            session=session,
            key=session_key or msg.session_key,
            raw=raw,
            loop=self,
        )
        if result := await self.commands.dispatch(command_ctx):
            return result

        # Handle slash commands
        cmd = raw.lower()
        cmd_token = cmd.split()[0] if cmd else ""
        cmd_name = cmd_token.split("@", 1)[0]
        if cmd_name == "/new":
            await self._await_inflight_consolidations()
            session = self._refresh_session(session)
            archive_messages = list(session.messages[session.last_consolidated :])
            if fixed_session_mode:
                temp_session = Session(
                    key=session.key,
                    conversation_key=session.conversation_key,
                )
                temp_session.messages = archive_messages
                temp_session.last_consolidated = 0
                archive_ok = await self._run_serialized_consolidation(
                    temp_session,
                    archive_all=True,
                )
                if not archive_ok:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Failed to archive fixed session history. Session was left unchanged.",
                    )
                session.metadata.pop("app_server_thread_id", None)
                session.metadata.pop("app_server_last_turn_id", None)
                session.clear()
                self.sessions.save(session)
                self._consolidation_scheduled_counts.pop(session.key, None)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                content="Cleared fixed session history.",
                )

            if self._uses_app_server_runtime():
                created = self.sessions.create_session(conversation_key, switch_to=True)
                content = (
                    f"New session started. Switched to session {created['id']}. "
                    "The previous session remains available in local session artifacts."
                )
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                )

            temp_session = Session(
                key=session.key,
                conversation_key=session.conversation_key,
            )
            temp_session.messages = archive_messages
            temp_session.last_consolidated = 0
            archive_ok = await self._run_serialized_consolidation(
                temp_session,
                archive_all=True,
            )
            if not archive_ok:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Failed to archive current session. New session was not created.",
                )

            created = self.sessions.create_session(conversation_key, switch_to=True)
            content = f"New session started. Switched to session {created['id']}."
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
            )
        if cmd_name == "/rebase":
            if not self._uses_app_server_runtime():
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Rebase is only available when nanobot is using Codex App Server.",
                )
            session = self._refresh_session(session)
            had_remote_thread = bool(str(session.metadata.get("app_server_thread_id") or "").strip())
            session.metadata.pop("app_server_thread_id", None)
            session.metadata.pop("app_server_last_turn_id", None)
            self.sessions.save(session)
            if had_remote_thread:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=(
                        "Cleared the current App Server thread binding. "
                        "The next turn will start a fresh Codex thread from local working set and recent history."
                    ),
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="This session is already detached from any remote App Server thread.",
            )
        if cmd_name == "/session":
            if fixed_session_mode:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Session management commands are unavailable in fixed-session mode.",
                )
            parts = msg.content.strip().split()
            if len(parts) >= 2 and parts[1].lower() == "list":
                snapshot = self.sessions.list_conversation_sessions(conversation_key)
                lines = ["Sessions:"]
                for item in snapshot["sessions"]:
                    marker = "*" if item["id"] == snapshot["active_session_id"] else "-"
                    title = str(item.get("title") or "(untitled)")
                    lines.append(f"{marker} {item['id']} {title}")
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="\n".join(lines),
                )
            if len(parts) >= 3 and parts[1].lower() == "switch":
                try:
                    switched = self.sessions.switch_session(conversation_key, parts[2])
                except ValueError as exc:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=str(exc),
                    )
                title = str(switched.get("title") or "(untitled)")
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Switched to session {switched['id']} ({title}).",
                )
            if len(parts) >= 2 and parts[1].lower() == "new":
                created = self.sessions.create_session(conversation_key, switch_to=True)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Created and switched to session {created['id']}.",
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Usage: /session list | /session new | /session switch <id>",
            )
        if cmd_name == "/help":
            if fixed_session_mode:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=(
                        "nanobot fixed-session mode:\n"
                        "/new - Clear current fixed session history\n"
                        "/rebase - Start a fresh Codex thread for the current fixed session\n"
                        "/help - Show available commands\n"
                        "/approve - Approve pending privileged request\n"
                        "/deny - Deny pending privileged request"
                    ),
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "nanobot commands:\n"
                    "/new - Start a new session in this conversation\n"
                    "/rebase - Start a fresh Codex thread for the current session\n"
                    "/session list - List sessions in this conversation\n"
                    "/session new - Create a new session\n"
                    "/session switch <id> - Switch active session\n"
                    "/help - Show available commands\n"
                    "/approve - Approve pending privileged request\n"
                    "/deny - Deny pending privileged request"
                ),
            )
        if cmd_name == "/approve":
            return await self._handle_privileged_approval(
                msg=msg,
                session=session,
                approval_key=lookup_session_key,
                approve=True,
            )
        if cmd_name == "/deny":
            return await self._handle_privileged_approval(
                msg=msg,
                session=session,
                approval_key=lookup_session_key,
                approve=False,
            )

        if not self._uses_app_server_runtime():
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        request_intent, request_execution, request_reason = await self._classify_request(
            session, msg.content
        )
        logger.info(
            f"Request routing for {msg.channel}:{msg.chat_id}: "
            f"intent={request_intent}, execution={request_execution} ({request_reason})"
        )
        if request_execution == self._REQUEST_EXEC_REQUIRED and not self._has_task_execution_tools():
            final_content = self._REQUIRED_EXEC_NO_TOOL_RESPONSE
            session.add_message("user", msg.content)
            session.add_message("assistant", final_content)
            session.metadata["last_request_reason"] = request_reason
            session.metadata["last_request_intent"] = request_intent
            session.metadata["last_request_execution"] = request_execution
            self.sessions.save(session)
            outbound_metadata = self._merge_outbound_metadata(
                msg.metadata,
                {
                    "request_intent": request_intent,
                    "request_execution": request_execution,
                },
            )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content,
                metadata=outbound_metadata,
            )

        self._set_tool_context(
            msg.channel,
            msg.chat_id,
            msg.sender_id,
            msg.metadata.get("message_id"),
            lookup_session_key=lookup_session_key,
            session=session,
        )
        async def _bus_progress(content: str = "", *, tool_hint: str | None = None) -> None:
            progress_text = self._strip_think(content)
            hint_text = self._strip_think(tool_hint)
            if not progress_text and not hint_text:
                return

            if progress_text:
                progress_metadata = dict(msg.metadata or {})
                progress_metadata["is_progress_update"] = True
                progress_metadata["_progress"] = True
                progress_metadata["_tool_hint"] = False
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=progress_text,
                        metadata=progress_metadata,
                    )
                )
                return

            hint_metadata = dict(msg.metadata or {})
            hint_metadata["is_progress_update"] = True
            hint_metadata["_progress"] = False
            hint_metadata["_tool_hint"] = True
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=hint_text or "",
                    metadata=hint_metadata,
                )
            )

        progress_cb = on_progress or _bus_progress

        if self._uses_app_server_runtime():
            final_content, tools_used, llm_metadata = await self._run_app_server_primary_turn(
                session=session,
                msg=msg,
                request_execution=request_execution,
                on_progress=progress_cb,
            )
        else:
            initial_messages = self.context.build_messages(
                history=session.get_history(max_messages=self.memory_window),
                current_message=msg.content,
                media=msg.media if msg.media else None,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )

            final_content, tools_used, llm_metadata = await self._run_agent_loop(
                initial_messages,
                request_execution=request_execution,
                on_progress=progress_cb,
            )
        llm_metadata["request_intent"] = request_intent
        llm_metadata["request_execution"] = request_execution
        session_trace_messages = llm_metadata.pop(self._SESSION_TRACE_MESSAGES_KEY, None)
        skip_session_assistant = bool(llm_metadata.pop(self._SKIP_SESSION_ASSISTANT_KEY, False))

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        else:
            final_content = self._strip_think(final_content) or final_content

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        session.add_message("user", msg.content)
        self._append_session_trace_messages(session, session_trace_messages)
        if not skip_session_assistant:
            session.add_message(
                "assistant",
                final_content,
                tools_used=tools_used if tools_used else None,
            )
        session.metadata["last_request_reason"] = request_reason
        session.metadata["last_request_intent"] = request_intent
        session.metadata["last_request_execution"] = request_execution
        self.sessions.save(session)

        outbound_metadata = self._merge_outbound_metadata(msg.metadata, llm_metadata)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=outbound_metadata,  # Pass through for channels and include LLM traces.
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

        conversation_key = f"{origin_channel}:{origin_chat_id}"
        session: Session | None = None
        if session_id := str(msg.metadata.get("session_id") or "").strip():
            session = self.sessions.get_by_id(session_id)
        if session is None and (session_key := str(msg.metadata.get("session_key") or "").strip()):
            session = self.sessions.get_or_create(session_key)
        if session is None:
            session = self.sessions.get_active_session(conversation_key)
        self._set_tool_context(
            origin_channel,
            origin_chat_id,
            msg.sender_id,
            msg.metadata.get("message_id"),
            lookup_session_key=self._task_session_key(session),
            session=session,
        )
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _, llm_metadata = await self._run_agent_loop(initial_messages)
        session_trace_messages = llm_metadata.pop(self._SESSION_TRACE_MESSAGES_KEY, None)
        skip_session_assistant = bool(llm_metadata.pop(self._SKIP_SESSION_ASSISTANT_KEY, False))

        if final_content is None:
            final_content = "Background task completed."
        else:
            final_content = self._strip_think(final_content) or final_content

        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        self._append_session_trace_messages(session, session_trace_messages)
        if not skip_session_assistant:
            session.add_message(
                "assistant",
                final_content,
            )
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content,
            metadata=self._merge_outbound_metadata(msg.metadata, llm_metadata),
        )

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
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
                return True

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(
                    f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})"
                )
                return True

            old_messages = session.messages[session.last_consolidated : -keep_count]
            if not old_messages:
                return True
            logger.info(
                f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep"
            )

        lines = []
        for m in old_messages:
            if str(m.get("role", "")).lower() == "tool":
                continue
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
            response = await self._provider_chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory consolidation agent. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return True
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(
                    f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}"
                )
                return True

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
            if getattr(session, "id", None):
                self.sessions.save(session)
            return True
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return False

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
            on_progress: Optional callback for intermediate progress output.

        Returns:
            The agent's outbound response payload.
        """
        return await self.process_direct_message(
            content=content,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            on_progress=on_progress,
        )

    async def process_direct_message(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the full outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)

        async with self._process_lock:
            kwargs: dict[str, Any] = {
                "session_key": session_key,
            }
            if on_progress is not None:
                kwargs["on_progress"] = on_progress
            try:
                return await self._process_message(msg, **kwargs)
            except TypeError as exc:
                if "on_progress" not in str(exc) or "on_progress" not in kwargs:
                    raise
                kwargs.pop("on_progress", None)
                return await self._process_message(msg, **kwargs)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message with per-session serialization and tracked task cleanup."""
        task = asyncio.current_task()
        if task is not None:
            self._active_tasks.setdefault(msg.session_key, []).append(task)
        try:
            lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
            if self._concurrency_gate is None:
                async with lock:
                    async with self._process_lock:
                        response = await self._process_message(msg)
            else:
                async with lock, self._concurrency_gate:
                    async with self._process_lock:
                        response = await self._process_message(msg)
            if response:
                await self.bus.publish_outbound(response)
        except asyncio.CancelledError:
            logger.info("Task cancelled for session {}", msg.session_key)
            raise
        except Exception as e:
            logger.error("Error processing message for session {}: {}", msg.session_key, e)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Sorry, I encountered an error: {e}",
                )
            )
        finally:
            if task is not None and (tasks := self._active_tasks.get(msg.session_key)):
                self._active_tasks[msg.session_key] = [item for item in tasks if item is not task]
                if not self._active_tasks[msg.session_key]:
                    self._active_tasks.pop(msg.session_key, None)
