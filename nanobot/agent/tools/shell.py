"""Shell execution tool."""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.security.approval_store import ApprovalStore
from nanobot.security.privileged_actions import parse_privileged_command


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        privileged_enabled: bool = False,
        approval_store: ApprovalStore | None = None,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",  # del /f, del /q
            r"\brmdir\s+/s\b",  # rmdir /s
            r"\b(format|mkfs|diskpart)\b",  # disk operations
            r"\bdd\s+if=",  # dd
            r">\s*/dev/sd",  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.privileged_enabled = privileged_enabled
        self.approval_store = approval_store
        self._default_channel = ""
        self._default_chat_id = ""
        self._default_sender_id = ""
        self._default_session_key = ""

    def set_context(
        self,
        channel: str,
        chat_id: str,
        sender_id: str = "",
        session_key: str = "",
    ) -> None:
        """Set current routing context for approval-gated privileged requests."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_sender_id = sender_id
        self._default_session_key = session_key

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            "Privileged commands require explicit user approval."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str, working_dir: str | None = None, **kwargs: Any) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        privileged = parse_privileged_command(command)
        if privileged.requires_approval:
            if privileged.error:
                return f"Error: {privileged.error}"
            return self._create_approval_request(
                command=command,
                working_dir=cwd,
                action=privileged.action or "",
                action_args=privileged.action_args,
            )

        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        return await self._run_command(command, cwd)

    async def _run_command(self, command: str, cwd: str) -> str:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                process.kill()
                return f"Error: Command timed out after {self.timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate very long output
            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _create_approval_request(
        self,
        *,
        command: str,
        working_dir: str,
        action: str,
        action_args: dict[str, Any],
    ) -> str:
        if os.name != "posix":
            return "Error: Privileged execution is supported only on Unix/Linux."

        if not self.privileged_enabled or not self.approval_store:
            return (
                "Error: Privileged command detected but privileged execution is not set up. "
                "Run `nanobot privileged setup` once, then retry."
            )

        if not self._default_channel or not self._default_chat_id:
            return "Error: Missing chat context for approval-gated privileged execution."

        session_key = f"{self._default_channel}:{self._default_chat_id}"
        if pending := self.approval_store.get_pending(session_key):
            return json.dumps(
                {
                    "approval_required": True,
                    "pending": True,
                    "request_id": pending.request_id,
                    "action": pending.action,
                    "message": (
                        "A privileged request is already pending in this chat. "
                        "Ask user to run /approve or /deny."
                    ),
                },
                ensure_ascii=False,
            )

        req = self.approval_store.create_pending(
            session_key=session_key,
            origin_session_key=self._default_session_key or None,
            channel=self._default_channel,
            chat_id=self._default_chat_id,
            requester_id=self._default_sender_id,
            command=command,
            working_dir=working_dir,
            action=action,
            action_args=action_args,
        )

        return json.dumps(
            {
                "approval_required": True,
                "pending": True,
                "request_id": req.request_id,
                "action": req.action,
                "action_args": req.action_args,
                "message": (
                    "Privileged execution requires explicit user approval. "
                    "Ask user to run /approve or /deny in this chat."
                ),
            },
            ensure_ascii=False,
        )

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            win_paths = re.findall(r"[A-Za-z]:\\[^\\\"']+", cmd)
            # Only match absolute paths — avoid false positives on relative
            # paths like ".venv/bin/python" where "/bin/python" would be
            # incorrectly extracted by the old pattern.
            posix_paths = re.findall(r"(?:^|[\s|>])(/[^\s\"'>]+)", cmd)

            for raw in win_paths + posix_paths:
                try:
                    p = Path(raw.strip()).resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None
