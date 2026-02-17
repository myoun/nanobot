"""Progress reporting tool for sending text updates to the current user chat."""

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class ReportToUserTool(Tool):
    """Send intermediate text updates to the current active chat."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set routing context for the current chat."""
        self._default_channel = channel
        self._default_chat_id = chat_id

    def set_send_callback(
        self,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """Set callback used to dispatch outbound messages."""
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "report_to_user"

    @property
    def description(self) -> str:
        return (
            "Send an intermediate text update to the current active user chat. "
            "Use this for progress reports, blockers, or clarification requests."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text to send to the current user chat.",
                    "minLength": 1,
                },
                "channel": {
                    "type": "string",
                    "description": "Optional override for target channel.",
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional override for target chat ID.",
                },
                "keep_typing": {
                    "type": "boolean",
                    "description": (
                        "Whether channels that support typing indicators should keep "
                        "showing typing after this progress update. Defaults to true."
                    ),
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        keep_typing: bool = True,
        **kwargs: Any,
    ) -> str:
        target_channel = (channel or self._default_channel).strip()
        target_chat_id = (chat_id or self._default_chat_id).strip()

        if not target_channel or not target_chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        text = content.strip()
        if not text:
            return "Error: content must not be empty"

        msg = OutboundMessage(
            channel=target_channel,
            chat_id=target_chat_id,
            content=text,
            metadata={
                "is_progress_update": True,
                "keep_typing": bool(keep_typing),
            },
        )

        try:
            await self._send_callback(msg)
            return f"Progress update sent to {target_channel}:{target_chat_id}"
        except Exception as e:
            return f"Error sending progress update: {str(e)}"
