"""Privileged action parsing (approval-gated, generic execution)."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


_PRIV_HINT_RE = re.compile(r"\b(sudo|apt-get|apt|systemctl|service|chown|chmod|mount|umount)\b")


@dataclass
class PrivilegedDecision:
    requires_approval: bool
    action: str | None = None
    action_args: dict = field(default_factory=dict)
    error: str | None = None


def parse_privileged_command(command: str) -> PrivilegedDecision:
    """Detect whether a command should go through privileged approval flow.

    This intentionally does not maintain an allowlist. Any privileged-looking
    command is routed to the approval-gated runner.
    """
    text = command.strip()
    if not text:
        return PrivilegedDecision(requires_approval=False)

    if not _PRIV_HINT_RE.search(text):
        return PrivilegedDecision(requires_approval=False)

    normalized = _normalize_command(text)
    if not normalized:
        return PrivilegedDecision(
            requires_approval=True,
            error="Unsupported privileged command: empty command after normalization.",
        )

    return PrivilegedDecision(
        requires_approval=True,
        action="shell_command",
        action_args={"command": normalized},
    )


def _normalize_command(command: str) -> str:
    """Strip only a leading sudo token; keep shell operators unchanged."""
    raw = command.strip()
    if not raw:
        return ""

    # Keep shell syntax intact (&&, |, redirects, etc.) and only trim the
    # leading "sudo " prefix if present.
    return re.sub(r"^\s*sudo\s+", "", raw, count=1).strip()
