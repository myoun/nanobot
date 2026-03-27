"""Minimal command routing table for slash commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from nanobot.bus.events import InboundMessage, OutboundMessage
    from nanobot.session.manager import Session

Handler = Callable[["CommandContext"], Awaitable["OutboundMessage | None"]]


@dataclass
class CommandContext:
    """Everything a command handler needs to produce a response."""

    msg: InboundMessage
    session: Session | None
    key: str
    raw: str
    args: str = ""
    loop: Any = None


class CommandRouter:
    """Pure dict-based command dispatch.

    Three tiers checked in order:
      1. *priority* — exact-match commands handled before the dispatch lock
         (e.g. /stop, /restart).
      2. *exact* — exact-match commands handled inside the dispatch lock.
      3. *prefix* — longest-prefix-first match (e.g. "/team ").
      4. *interceptors* — fallback predicates (e.g. team-mode active check).
    """

    def __init__(self) -> None:
        self._priority: dict[str, Handler] = {}
        self._exact: dict[str, Handler] = {}
        self._prefix: list[tuple[str, Handler]] = []
        self._interceptors: list[Handler] = []

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize Telegram-style /command@bot syntax to plain /command."""
        raw = text.strip()
        if not raw:
            return raw

        parts = raw.split(maxsplit=1)
        head = parts[0]
        tail = parts[1] if len(parts) > 1 else ""

        if head.startswith("/") and "@" in head:
            head = head.split("@", 1)[0]

        normalized = head
        if tail:
            normalized += f" {tail}"
        return normalized

    def priority(self, cmd: str, handler: Handler) -> None:
        self._priority[cmd] = handler

    def exact(self, cmd: str, handler: Handler) -> None:
        self._exact[cmd] = handler

    def prefix(self, pfx: str, handler: Handler) -> None:
        self._prefix.append((pfx, handler))
        self._prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def intercept(self, handler: Handler) -> None:
        self._interceptors.append(handler)

    def is_priority(self, text: str) -> bool:
        return self._normalize(text).lower() in self._priority

    async def dispatch_priority(self, ctx: CommandContext) -> OutboundMessage | None:
        """Dispatch a priority command. Called from run() without the lock."""
        normalized = self._normalize(ctx.raw)
        handler = self._priority.get(normalized.lower())
        if handler:
            ctx.raw = normalized
            return await handler(ctx)
        return None

    async def dispatch(self, ctx: CommandContext) -> OutboundMessage | None:
        """Try exact, prefix, then interceptors. Returns None if unhandled."""
        ctx.raw = self._normalize(ctx.raw)
        cmd = ctx.raw.lower()

        if handler := self._exact.get(cmd):
            return await handler(ctx)

        for pfx, handler in self._prefix:
            if cmd.startswith(pfx):
                ctx.args = ctx.raw[len(pfx):]
                return await handler(ctx)

        for interceptor in self._interceptors:
            result = await interceptor(ctx)
            if result is not None:
                return result

        return None
