"""Helpers for routing cron deliveries to channel-specific targets."""

from typing import Any


def resolve_delivery_target(channel: str, target: str | None) -> tuple[str, dict[str, Any]]:
    """Return a chat ID plus outbound metadata for a stored cron target."""
    if not target:
        return "direct", {}

    if channel != "telegram":
        return target, {}

    raw_target = target
    if raw_target.startswith("telegram:"):
        raw_target = raw_target.removeprefix("telegram:")

    if ":topic:" not in raw_target:
        return raw_target, {}

    chat_id, thread_id = raw_target.rsplit(":topic:", 1)
    if not chat_id or not thread_id.isdigit():
        return target, {}

    return chat_id, {"message_thread_id": int(thread_id)}
