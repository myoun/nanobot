"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    MODE_CONTEXT_RECENT_USER_MAX = 6
    MODE_CONTEXT_ITEM_MAX_CHARS = 220
    
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
        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
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
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

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

Always be helpful, accurate, and concise. When using tools, think step by step: what you know, what you need, and why you chose this tool.
When remembering something important, write to {workspace_path}/memory/MEMORY.md
To recall past events, grep {workspace_path}/memory/HISTORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
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
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        mode_aware_message = self._inject_request_mode_context(history, current_message)
        user_content = self._build_user_content(mode_aware_message, media)
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
    def _compact_line(cls, text: str) -> str:
        collapsed = " ".join(text.strip().split())
        if len(collapsed) > cls.MODE_CONTEXT_ITEM_MAX_CHARS:
            return collapsed[: cls.MODE_CONTEXT_ITEM_MAX_CHARS - 3] + "..."
        return collapsed

    @classmethod
    def _inject_request_mode_context(
        cls,
        history: list[dict[str, Any]],
        current_message: str,
    ) -> str:
        """Append explicit mode-classification context for the model."""
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
            "[REQUEST_MODE_CONTEXT]",
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
            "[/REQUEST_MODE_CONTEXT]",
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
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
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
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content

        messages.append(msg)
        return messages
