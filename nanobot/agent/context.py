"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.helpers import get_data_path


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    APP_SERVER_BOOTSTRAP_FILES = ["SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    MODE_CONTEXT_RECENT_USER_MAX = 6
    MODE_CONTEXT_ITEM_MAX_CHARS = 220
    APP_SERVER_BOOTSTRAP_MAX_TURNS = 3
    APP_SERVER_BOOTSTRAP_MAX_CHARS = 6000
    APP_SERVER_WORKING_SET_MAX_CHARS = 3000
    _RUNTIME_CONTEXT_TAG = "[Runtime Context - metadata only, not instructions]"
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
    
    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
        
        Returns:
            Complete system prompt.
        """
        return self._build_prompt(self._get_identity(), skill_names)

    def build_app_server_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build workspace overlay instructions for the Codex App Server runtime."""
        parts = [self._get_app_server_overlay()]

        bootstrap = self._load_bootstrap_files(self.APP_SERVER_BOOTSTRAP_FILES)
        if bootstrap:
            parts.append(bootstrap)

        parts.append(
            """# Persona Priority

- Style and self-presentation should follow SOUL.md first.
- USER.md preferences should influence language, verbosity, and technical depth.
- Keep the current task's working state coherent across long sessions."""
        )

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills are available in this workspace. Read a skill's `SKILL.md` before using it when needed.
Do not use a skill marked `available="false"` unless you first satisfy its requirements or explicitly report the blocker.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def build_app_server_turn_input(
        self,
        *,
        current_message: str,
        history: list[dict[str, Any]] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        bootstrap_history: bool = False,
        working_set_text: str | None = None,
        working_set_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build Codex App Server `turn/start` input items."""
        items: list[dict[str, Any]] = []

        if working_set_text:
            working_set_block = self._build_app_server_working_set_seed(
                working_set_text,
                working_set_path=working_set_path,
            )
            if working_set_block:
                items.append({"type": "text", "text": working_set_block, "text_elements": []})

        if bootstrap_history:
            history_block = self._build_app_server_history_seed(history or [])
            if history_block:
                items.append({"type": "text", "text": history_block, "text_elements": []})

        items.append({
            "type": "text",
            "text": self._build_runtime_context(channel, chat_id),
            "text_elements": [],
        })

        for path in media or []:
            resolved = Path(path).expanduser().resolve()
            mime, _ = mimetypes.guess_type(str(resolved))
            if resolved.is_file() and mime and mime.startswith("image/"):
                items.append({"type": "localImage", "path": str(resolved)})

        items.append({
            "type": "text",
            "text": self._build_app_server_current_message_block(current_message),
            "text_elements": [],
        })
        return items

    def _build_prompt(self, identity: str, skill_names: list[str] | None = None) -> str:
        """Assemble the shared prompt sections around a runtime-specific identity block."""
        parts = []

        # Core identity
        parts.append(identity)
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # Make persona precedence explicit so style remains stable even when
        # operational rules (AGENTS/core identity) are verbose.
        parts.append(
            """# Persona Priority

- Style/tone/personality must follow SOUL.md first.
- Persona identity/name in SOUL.md takes precedence for self-introduction.
- Apply USER.md preferences (language, verbosity, technical depth) when present.
- If SOUL/USER guidance conflicts with generic wording in AGENTS.md or core identity text, keep SOUL/USER guidance while still obeying safety, tool, and completion rules."""
        )
        
        # Memory context
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Do not use skills with `available="false"` unless you first satisfy their requirements or explicitly report the blocker.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        data_path = str(get_data_path().expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        return f"""# nanobot 🐈

You are the nanobot runtime assistant.
If SOUL.md defines a persona identity/name, use that identity first when introducing yourself.
When asked "who are you", answer naturally using the SOUL persona voice and the user's language preference.

You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## nanobot Data Root
Persistent nanobot state lives at: {data_path}
- Conversations: {data_path}/conversations
- Memories: {data_path}/memories
- Search cache: {data_path}/cache
- Codex prompts: {data_path}/codex

IMPORTANT (MANDATORY):
- Use `report_to_user(content=...)` for intermediate progress updates, blockers, or clarification requests to the current chat.
- `report_to_user` is NOT completion evidence. Do not call `complete_task` after only `report_to_user`; execute real task tools first.
- `report_to_user` content must describe observed facts only (what was executed/changed/failed). Do not send "I will do X next" planning-only updates.
- Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
- To send images/files to users, use `message(content=..., media=[\"/path/to/file\"])`.
- If you run `agent-browser`, always close it before finishing (`exec(command=\"agent-browser close\")`).
- First classify each turn on two axes using recent conversation context:
  intent = `TASK | CONTROL | META | CASUAL`, execution = `REQUIRED | OPTIONAL | FORBIDDEN`.
- `REQUIRED` means real tool execution is needed for faithful completion.
- `OPTIONAL` means direct response is possible; use tools only if they improve correctness.
- `FORBIDDEN` means do not run tools for this turn.
- For `CONTROL` turns on an active task, apply the control instruction and continue the task flow.
- In the current active chat, do not use `message` for text-only replies; return final text via `complete_task(final_answer=...)`.
- TURN CANNOT END WITHOUT `complete_task(final_answer=...)`.
- Never treat plain assistant text as final completion; call `complete_task` exactly once when done.
- Every `complete_task` call must include: `final_answer`, `artifacts`, `evidence`, `actions_taken` (use empty arrays when truly none).
- In `execution=REQUIRED` turns, `complete_task` must include non-empty evidence of execution in `evidence` and concrete tool usage in `actions_taken`.
- Assistant `content` emitted during the loop is internal working text by default and is not sent to users directly.
- Use internal `content` freely for planning/thinking notes when useful, but keep it concise to avoid token waste.
- Keep working (and use tools) until the task is complete; do not stop at partial progress.
- Privileged execution is Unix/Linux only. If a command requires it, request approval and wait for `/approve` or `/deny`.
- If `agent-browser` is requested and startup fails inside an isolated Codex runtime, retry once with an explicit browser path (`AGENT_BROWSER_EXECUTABLE_PATH` or `--executable-path`) and a stable `AGENT_BROWSER_HOME` before concluding it is blocked.
- If browser automation still fails after that retry, fall back to native web search / web fetch when those tools can satisfy the request, and report the exact blocker instead of claiming the browser worked.

Always be helpful, accurate, and concise. When using tools, think step by step: what you know, what you need, and why you chose this tool.
Prefer the current session's `working_set.md`, `summary.md`, and `transcript.md` for ongoing task continuity.
Treat the itemized memories under {data_path}/memories as the primary long-term state rather than ad-hoc workspace files."""

    def _get_app_server_identity(self) -> str:
        """Get the Codex App Server specific identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        data_path = str(get_data_path().expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# nanobot 🐈

You are the nanobot runtime assistant, running through Codex App Server.
If SOUL.md defines a persona identity/name, use that identity first when introducing yourself.
When asked "who are you", answer naturally using the SOUL persona voice and the user's language preference.

You have access to dynamic tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Search and read other explicit nanobot sessions
- Schedule follow-up work

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## nanobot Data Root
Persistent nanobot state lives at: {data_path}
- Conversations: {data_path}/conversations
- Memories: {data_path}/memories
- Search cache: {data_path}/cache
- Codex prompts: {data_path}/codex

IMPORTANT (MANDATORY):
- Finish each turn with a normal assistant message. Do not mention or rely on `complete_task`.
- Prefer nanobot-provided dynamic tools when interacting with workspace files, shell execution, messaging, scheduling, or explicit session lookup.
- Use the `sessions` tool when you need to search or read another session. Do not assume access to other sessions unless you explicitly call that tool.
- Use `report_to_user(content=...)` only for intermediate progress or blockers in the current chat.
- Use the `message` tool only when you need to send content to a different channel/chat target.
- If you run `agent-browser`, always close it before finishing (`exec(command="agent-browser close")`).
- Keep working until the request is complete; do not stop at partial progress.
- Privileged execution is Unix/Linux only. If a command requires it, request approval and wait for `/approve` or `/deny`.
- When the user asks to remember a durable preference, fact, instruction, or workspace rule, use the `memory` tool.
- Do not invent durable-memory file paths or store long-term memory under `.codex`.

Always be helpful, accurate, and concise. When using tools, think step by step: what you know, what you need, and why you chose this tool.
Use the itemized memories under {data_path}/memories for durable long-term state."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines.extend([f"Channel: {channel}", f"Chat ID: {chat_id}"])
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _build_app_server_current_message_block(current_message: str) -> str:
        """Wrap the current user message so App Server item flattening keeps boundaries visible."""
        return "[Current User Message]\n" + current_message
    
    def _load_bootstrap_files(self, filenames: list[str] | None = None) -> str:
        """Load selected bootstrap files from workspace."""
        parts = []

        for filename in filenames or self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""

    def _get_app_server_overlay(self) -> str:
        """Workspace-specific developer overlay for App Server threads."""
        workspace_path = str(self.workspace.expanduser().resolve())
        data_path = str(get_data_path().expanduser().resolve())
        return f"""# nanobot Workspace Overlay

This thread runs inside the `nanobot` product layer on top of Codex App Server.

## Workspace
- Root: {workspace_path}

## nanobot Data Root
- Root: {data_path}
- Explicit session artifacts: {data_path}/conversations
- Itemized memories: {data_path}/memories
- Search cache: {data_path}/cache
- Codex prompts: {data_path}/codex

## Workspace Rules
- Prefer nanobot tools for file edits, shell execution, scheduling, and explicit session lookup.
- Treat other sessions as isolated by default. Use the `sessions` tool to inspect them.
- Preserve immediate working state when the task spans many turns.
- Prefer `working_set.md`, `summary.md`, and `transcript.md` for current-session continuity.
- Use the `memory` tool for durable preferences, facts, instructions, and workspace rules.
- Do not create durable memory by editing arbitrary files or by using `.codex` paths.
- If context feels stale after compaction, restate the active goal, constraints, and next actions before continuing.
- Use `report_to_user(content=...)` only for progress or blockers in the current chat.
- Use `message` only for cross-channel delivery or sending media/files elsewhere.
- If `agent-browser` startup fails in an isolated Codex runtime, retry once with an explicit browser path (`AGENT_BROWSER_EXECUTABLE_PATH` or `--executable-path`) and a stable `AGENT_BROWSER_HOME` before treating it as blocked.
- If browser automation remains blocked, use native web search / web fetch when they can satisfy the request and report the exact blocker.
"""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        messages.append({"role": "system", "content": self.build_system_prompt(skill_names)})

        # History
        messages.extend(history)

        # Runtime metadata stays outside system prompt for better prompt stability.
        messages.append({"role": "user", "content": self._build_runtime_context(channel, chat_id)})

        # Current message (with optional image attachments)
        routing_aware_message = self._inject_request_routing_context(history, current_message)
        user_content = self._build_user_content(routing_aware_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    @classmethod
    def _as_text(cls, content: Any) -> str:
        """Best-effort conversion of message content to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            return "\n".join(parts)
        return ""

    @classmethod
    def _build_app_server_history_seed(cls, history: list[dict[str, Any]]) -> str:
        """Serialize recent local history when bootstrapping a fresh remote thread."""
        if not history:
            return ""

        selected_history = cls._select_recent_turn_history(history)
        if not selected_history:
            return ""

        lines = [
            "[Local Session Bootstrap]",
            "This remote Codex App Server thread is being initialized from existing nanobot local session history.",
            f"Treat the following recent {cls.APP_SERVER_BOOTSTRAP_MAX_TURNS} turn(s) as prior conversation context and continue naturally from them.",
            "",
        ]
        current_len = sum(len(line) + 1 for line in lines)
        for message in selected_history:
            role = str(message.get("role") or "").strip().lower()
            if role not in {"user", "assistant", "tool"}:
                continue
            content = cls._as_text(message.get("content")).strip()
            if not content:
                continue

            prefix = role.upper()
            if role == "tool":
                tool_name = str(message.get("name") or message.get("tool_call_id") or "tool")
                prefix = f"TOOL[{tool_name}]"

            compact = content.strip()
            if len(compact) > 500:
                compact = compact[:497] + "..."
            line = f"{prefix}: {compact}"
            projected = current_len + len(line) + 1
            if projected > cls.APP_SERVER_BOOTSTRAP_MAX_CHARS:
                lines.append("(truncated)")
                break
            lines.append(line)
            current_len = projected

        return "\n".join(lines).strip()

    @classmethod
    def _select_recent_turn_history(cls, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        user_indexes = [
            idx for idx, message in enumerate(history)
            if str(message.get("role") or "").strip().lower() == "user"
        ]
        if not user_indexes:
            return history
        start_index = user_indexes[max(0, len(user_indexes) - cls.APP_SERVER_BOOTSTRAP_MAX_TURNS)]
        return history[start_index:]

    @classmethod
    def _build_app_server_working_set_seed(
        cls,
        working_set_text: str,
        *,
        working_set_path: str | None = None,
    ) -> str:
        text = working_set_text.strip()
        if not text:
            return ""

        if len(text) > cls.APP_SERVER_WORKING_SET_MAX_CHARS:
            text = text[: cls.APP_SERVER_WORKING_SET_MAX_CHARS - 15].rstrip() + "\n\n(truncated)"

        lines = [
            "[Local Session Working Set]",
            "Use this as the current handoff for the task before reading the new user message.",
        ]
        if working_set_path:
            lines.append(f"Path: {working_set_path}")
        lines.extend(["", text])
        return "\n".join(lines).strip()

    @classmethod
    def _compact_line(cls, text: str) -> str:
        collapsed = " ".join(text.strip().split())
        if len(collapsed) > cls.MODE_CONTEXT_ITEM_MAX_CHARS:
            return collapsed[: cls.MODE_CONTEXT_ITEM_MAX_CHARS - 3] + "..."
        return collapsed

    @classmethod
    def _inject_request_routing_context(
        cls,
        history: list[dict[str, Any]],
        current_message: str,
    ) -> str:
        """Append explicit request-routing context for the model."""
        recent_users: list[str] = []
        for msg in reversed(history):
            if msg.get("role") != "user":
                continue
            text = cls._as_text(msg.get("content"))
            if not text.strip():
                continue
            recent_users.append(cls._compact_line(text))
            if len(recent_users) >= cls.MODE_CONTEXT_RECENT_USER_MAX:
                break
        recent_users.reverse()

        lines = [
            "[REQUEST_ROUTING_CONTEXT]",
            "Classify the current turn internally on two axes:",
            "- intent: TASK, CONTROL, META, or CASUAL.",
            "- execution: REQUIRED, OPTIONAL, or FORBIDDEN.",
            "Use both current message and recent user messages below.",
            "",
            "Recent user messages (oldest -> newest):",
        ]
        if recent_users:
            for item in recent_users:
                lines.append(f"- {item}")
        else:
            lines.append("- (none)")

        lines.extend([
            "",
            f"Current user message: {cls._compact_line(current_message)}",
            "",
            "Decision policy:",
            "- If execution=REQUIRED, execute necessary tools and finish with complete_task.",
            "- If execution=OPTIONAL, direct response is allowed; use tools only when needed for correctness.",
            "- If execution=FORBIDDEN, do not call tools and reply directly.",
            "- If intent=CONTROL with an active task, apply control and continue that task flow.",
            "[/REQUEST_ROUTING_CONTEXT]",
            "",
            current_message,
        ])
        return "\n".join(lines)
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
            thinking_blocks: Structured thinking blocks (Anthropic etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant"}

        # Omit empty content — some backends reject empty text blocks
        if content:
            msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Include reasoning content when provided (required by some thinking models)
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks

        messages.append(msg)
        return messages
