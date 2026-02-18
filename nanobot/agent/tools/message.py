"""Message tool for sending messages to users."""

from typing import Any, Callable, Awaitable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""
    
    def __init__(
        self, 
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = ""
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
    
    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback
    
    @property
    def name(self) -> str:
        return "message"
    
    @property
    def description(self) -> str:
        return (
            "Send a message to the user. Supports text content and optional media file paths/URLs "
            "(channel-dependent). For the current active chat, use this primarily for media delivery."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text content to send (use empty string when sending only media)"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional media file paths or URLs to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                }
            },
            "required": ["content"]
        }
    
    async def execute(
        self, 
        content: str = "",
        media: list[str] | None = None,
        channel: str | None = None, 
        chat_id: str | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        
        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        
        if not self._send_callback:
            return "Error: Message sending not configured"

        content = content.strip()
        cleaned_media = [m.strip() for m in (media or []) if isinstance(m, str) and m.strip()]
        if not content and not cleaned_media:
            return "Error: Provide at least one of 'content' or 'media'"

        is_current_chat = (
            channel == self._default_channel and chat_id == self._default_chat_id
        )
        if is_current_chat and not cleaned_media:
            return (
                "Error: Text-only message to the current chat is blocked. "
                "Use complete_task(final_answer=...) for normal replies."
            )
        
        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=cleaned_media,
        )
        
        try:
            await self._send_callback(msg)
            if cleaned_media:
                return f"Message sent to {channel}:{chat_id} with {len(cleaned_media)} media item(s)"
            return f"Message sent to {channel}:{chat_id}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
