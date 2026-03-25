"""Workspace-local Codex profile management for nanobot."""

from __future__ import annotations

import os
from pathlib import Path
import re

from nanobot.utils.helpers import ensure_dir, get_data_path

_MANAGED_START = "# >>> nanobot managed profile >>>"
_MANAGED_END = "# <<< nanobot managed profile <<<"
_TRUST_MANAGED_START = "# >>> nanobot managed trusted project >>>"
_TRUST_MANAGED_END = "# <<< nanobot managed trusted project <<<"

DEFAULT_SYSTEM_PROMPT = """# nanobot Codex Runtime

You are running as the `nanobot` coding agent through Codex App Server.

## Operating Model

- Prefer direct, factual, concise answers.
- Treat the current session as the primary source of active task state.
- Other sessions are isolated by default. Only inspect them through explicit nanobot tools.
- Global memory, workspace rules, and automations are distinct scopes. Do not collapse them into generic session notes.

## Working State Priority

When context is large or noisy, prioritize preserving:

1. The current goal.
2. The files and modules currently in play.
3. Recent decisions and why they were made.
4. Unfinished TODOs, blockers, and next actions.
5. Constraints that must not be violated.

Do not turn an in-progress coding task into a generic retrospective summary.

## Tool Use

- Use nanobot-provided tools when available for file edits, shell execution, messaging, scheduling, and explicit session lookup.
- Use the `sessions` tool when you need to search or read another session. Never assume hidden cross-session access.
- Use the `memory` tool when the user asks to remember a durable preference, fact, instruction, or workspace rule.
- Never invent a durable-memory file path or store long-term memory under `.codex`.
- Keep tool use grounded in the current request; do not wander into unrelated work.

## Output

- Continue the active task until it is actually complete.
- Prefer concrete execution over vague planning when execution is possible.
- If context quality degrades, preserve and restate the active working state before continuing.
"""

DEFAULT_COMPACT_PROMPT = """# nanobot Compaction Handoff

Produce a compaction handoff that preserves the immediate working state of the current coding task.

## Required Sections

### Current Goal
- State the single task currently in progress.

### Active Files
- List the files, modules, or commands that matter right now.
- Explain briefly why each one matters.

### Recent Decisions
- Record concrete decisions or edits that shape the next step.

### Open TODOs
- List unfinished work items that still need to happen.

### Blockers and Hypotheses
- Capture active blockers, uncertainties, and working hypotheses.

### Constraints
- Include requirements that must not be violated.

### Next Actions
- Give the next 2-5 actions the agent should take immediately after compaction.

## Rules

- Do not write a generic retrospective.
- Do not optimize for completeness over continuity.
- Preserve what the next turn needs to continue without re-discovering the plan.
- Prefer concrete task state over broad background summary.
"""


class CodexProfileManager:
    """Ensures a workspace-local Codex profile exists for nanobot."""

    def __init__(self, workspace: Path, profile_name: str = "nanobot"):
        self.workspace = workspace.resolve()
        self.profile_name = profile_name
        self.project_codex_dir = ensure_dir(self.workspace / ".codex")
        self.project_config_file = self.project_codex_dir / "config.toml"
        codex_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex")).expanduser()
        self.codex_home = ensure_dir(codex_home)
        self.global_config_file = self.codex_home / "config.toml"
        self.nanobot_codex_dir = ensure_dir(get_data_path() / "codex")
        self.system_prompt_file = self.nanobot_codex_dir / "system_prompt.md"
        self.compact_prompt_file = self.nanobot_codex_dir / "compact_prompt.md"

    def ensure_profile(self) -> None:
        """Write the compact prompt and managed profile block if needed."""
        self._ensure_system_prompt()
        self._ensure_compact_prompt()
        self._ensure_project_profile_block()
        self._ensure_workspace_trust()

    def remove_managed_profile(self) -> bool:
        """Remove nanobot-managed profile material when disabling workspace profile usage."""
        changed = False

        if self.project_config_file.exists():
            existing = self.project_config_file.read_text(encoding="utf-8")
            if _MANAGED_START in existing and _MANAGED_END in existing:
                start = existing.index(_MANAGED_START)
                end = existing.index(_MANAGED_END, start) + len(_MANAGED_END)
                updated = existing[:start].rstrip()
                tail = existing[end:].lstrip()
                if updated and tail:
                    updated += "\n\n" + tail
                elif tail:
                    updated = tail
                updated = updated.rstrip()
                if updated:
                    updated += "\n"
                    self.project_config_file.write_text(updated, encoding="utf-8")
                else:
                    self.project_config_file.unlink(missing_ok=True)
                changed = True

        for path in (
            self.system_prompt_file,
            self.compact_prompt_file,
        ):
            if path.exists():
                path.unlink()
                changed = True

        return changed

    def _ensure_system_prompt(self) -> None:
        self._ensure_markdown_prompt(target=self.system_prompt_file, default_text=DEFAULT_SYSTEM_PROMPT)

    def _ensure_compact_prompt(self) -> None:
        self._ensure_markdown_prompt(target=self.compact_prompt_file, default_text=DEFAULT_COMPACT_PROMPT)

    @staticmethod
    def _ensure_markdown_prompt(*, target: Path, default_text: str) -> None:
        if target.exists():
            return
        target.write_text(default_text, encoding="utf-8")

    def _ensure_project_profile_block(self) -> None:
        managed_block = self._build_managed_block()
        if not self.project_config_file.exists():
            self.project_config_file.write_text(managed_block, encoding="utf-8")
            return

        existing = self.project_config_file.read_text(encoding="utf-8")
        profile_header = f"[profiles.{self.profile_name}]"

        if _MANAGED_START in existing and _MANAGED_END in existing:
            start = existing.index(_MANAGED_START)
            end = existing.index(_MANAGED_END, start) + len(_MANAGED_END)
            replacement = managed_block.rstrip()
            updated = existing[:start].rstrip()
            if updated:
                updated += "\n\n"
            updated += replacement
            tail = existing[end:].lstrip()
            if tail:
                updated += "\n\n" + tail
            updated = updated.rstrip() + "\n"
            if updated != existing:
                self.project_config_file.write_text(updated, encoding="utf-8")
            return

        if profile_header in existing:
            return

        content = existing.rstrip()
        if content:
            content += "\n\n"
        content += managed_block
        self.project_config_file.write_text(content.rstrip() + "\n", encoding="utf-8")

    def _build_managed_block(self) -> str:
        system_prompt_path = self.system_prompt_file.resolve()
        compact_prompt_path = self.compact_prompt_file.resolve()
        return (
            f"{_MANAGED_START}\n"
            f"[profiles.{self.profile_name}]\n"
            f'model_instructions_file = "{system_prompt_path}"\n'
            f'experimental_compact_prompt_file = "{compact_prompt_path}"\n'
            f"{_MANAGED_END}\n"
        )

    def _ensure_workspace_trust(self) -> None:
        header = f'[projects."{self.workspace}"]'
        trust_line = 'trust_level = "trusted"'

        if not self.global_config_file.exists():
            self.global_config_file.write_text(
                f"{_TRUST_MANAGED_START}\n{header}\n{trust_line}\n{_TRUST_MANAGED_END}\n",
                encoding="utf-8",
            )
            return

        existing = self.global_config_file.read_text(encoding="utf-8")
        section_pattern = re.compile(
            rf"(?ms)^(?P<header>\[projects\.\"{re.escape(str(self.workspace))}\"\]\s*\n)(?P<body>.*?)(?=^\[|\Z)"
        )
        match = section_pattern.search(existing)
        if match:
            body = match.group("body")
            if re.search(r'(?m)^trust_level\s*=\s*"trusted"\s*$', body):
                return
            if re.search(r"(?m)^trust_level\s*=", body):
                updated_body = re.sub(
                    r'(?m)^trust_level\s*=\s*".*?"\s*$',
                    trust_line,
                    body,
                    count=1,
                )
            else:
                updated_body = trust_line + ("\n" + body if body else "\n")
            updated = existing[: match.start("body")] + updated_body + existing[match.end("body") :]
            self.global_config_file.write_text(updated, encoding="utf-8")
            return

        managed_block = f"{_TRUST_MANAGED_START}\n{header}\n{trust_line}\n{_TRUST_MANAGED_END}\n"
        content = existing.rstrip()
        if content:
            content += "\n\n"
        content += managed_block
        self.global_config_file.write_text(content.rstrip() + "\n", encoding="utf-8")
