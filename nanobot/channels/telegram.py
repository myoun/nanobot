"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import mimetypes
import re
from pathlib import Path
from typing import Any

from loguru import logger
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import TelegramConfig
from nanobot.session.manager import SessionManager


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""

    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []

    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", save_code_block, text)

    # 2. Extract and protect inline code
    inline_codes: list[str] = []

    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", save_inline_code, text)

    # 3. Headers # Title -> just the title text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r"^>\s*(.*)$", r"\1", text, flags=re.MULTILINE)

    # 5. Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # 7. Bold **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"<i>\1</i>", text)

    # 9. Strikethrough ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # 10. Bullet lists - item -> • item
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)

    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")

    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")

    return text


def _split_message(content: str, max_len: int = 4000) -> list[str]:
    """Split content into chunks within max_len, preferring line breaks."""
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        pos = cut.rfind("\n")
        if pos == -1:
            pos = cut.rfind(" ")
        if pos == -1:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling.

    Simple and reliable - no webhook/public IP needed.
    """

    name = "telegram"
    _SESSION_CB_PREFIX = "nbs"
    _SESSION_PAGE_SIZE = 6
    _SESSION_TITLE_MAX_LEN = 28

    # Commands registered with Telegram's command menu
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("sessions", "List sessions in this chat"),
        BotCommand("session", "Manage sessions"),
        BotCommand("help", "Show available commands"),
        BotCommand("approve", "Approve pending privileged request"),
        BotCommand("deny", "Deny pending privileged request"),
    ]

    def __init__(
        self,
        config: TelegramConfig,
        bus: MessageBus,
        groq_api_key: str = "",
        session_manager: SessionManager | None = None,
    ):
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self.groq_api_key = groq_api_key
        self._session_manager = session_manager
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._typing_tasks: dict[str, asyncio.Task] = {}  # chat_id -> typing loop task

    @classmethod
    def _short_session_title(cls, title: str) -> str:
        raw = " ".join((title or "").split()).strip()
        if not raw:
            raw = "New chat"
        if len(raw) <= cls._SESSION_TITLE_MAX_LEN:
            return raw
        return raw[: cls._SESSION_TITLE_MAX_LEN - 1].rstrip() + "..."

    @staticmethod
    def _conversation_key(chat_id: str) -> str:
        return f"telegram:{chat_id}"

    @classmethod
    def _session_pages(cls, total_items: int) -> int:
        if total_items <= 0:
            return 1
        return (total_items + cls._SESSION_PAGE_SIZE - 1) // cls._SESSION_PAGE_SIZE

    def _build_sessions_keyboard(self, snapshot: dict[str, Any], page: int) -> InlineKeyboardMarkup:
        sessions = snapshot.get("sessions")
        if not isinstance(sessions, list):
            sessions = []

        total_pages = self._session_pages(len(sessions))
        normalized_page = min(max(0, page), total_pages - 1)
        start = normalized_page * self._SESSION_PAGE_SIZE
        end = start + self._SESSION_PAGE_SIZE
        visible = sessions[start:end]

        rows: list[list[InlineKeyboardButton]] = []
        for item in visible:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or "").strip()
            if not sid:
                continue
            marker = "●" if bool(item.get("active")) else "○"
            pin = "📌 " if bool(item.get("pinned", False)) else ""
            label = f"{marker} {pin}{self._short_session_title(str(item.get('title') or ''))}"
            rows.append(
                [
                    InlineKeyboardButton(
                        label,
                        callback_data=(f"{self._SESSION_CB_PREFIX}:sw:{sid}:{normalized_page}"),
                    )
                ]
            )

        rows.append(
            [
                InlineKeyboardButton(
                    "➕ New",
                    callback_data=f"{self._SESSION_CB_PREFIX}:new:{normalized_page}",
                ),
                InlineKeyboardButton(
                    "📌 Pin",
                    callback_data=f"{self._SESSION_CB_PREFIX}:pin:active:{normalized_page}",
                ),
                InlineKeyboardButton(
                    "📍 Unpin",
                    callback_data=f"{self._SESSION_CB_PREFIX}:unpin:active:{normalized_page}",
                ),
            ]
        )
        rows.append(
            [
                InlineKeyboardButton(
                    "🔄 Refresh",
                    callback_data=f"{self._SESSION_CB_PREFIX}:rf:{normalized_page}",
                ),
                InlineKeyboardButton(
                    "🔎 Search",
                    callback_data=f"{self._SESSION_CB_PREFIX}:helpsearch:{normalized_page}",
                ),
            ]
        )

        if total_pages > 1:
            nav_row: list[InlineKeyboardButton] = []
            if normalized_page > 0:
                nav_row.append(
                    InlineKeyboardButton(
                        "◀ Prev",
                        callback_data=(f"{self._SESSION_CB_PREFIX}:pg:{normalized_page - 1}"),
                    )
                )
            nav_row.append(
                InlineKeyboardButton(
                    f"{normalized_page + 1}/{total_pages}",
                    callback_data=f"{self._SESSION_CB_PREFIX}:noop",
                )
            )
            if normalized_page < total_pages - 1:
                nav_row.append(
                    InlineKeyboardButton(
                        "Next ▶",
                        callback_data=(f"{self._SESSION_CB_PREFIX}:pg:{normalized_page + 1}"),
                    )
                )
            rows.append(nav_row)

        return InlineKeyboardMarkup(rows)

    @classmethod
    def _render_sessions_panel_text(
        cls,
        snapshot: dict[str, Any],
        *,
        page: int,
        notice: str | None = None,
    ) -> str:
        sessions = snapshot.get("sessions")
        if not isinstance(sessions, list):
            sessions = []
        total_pages = cls._session_pages(len(sessions))
        normalized_page = min(max(0, page), total_pages - 1)
        active = next(
            (item for item in sessions if isinstance(item, dict) and bool(item.get("active"))),
            None,
        )
        active_title = cls._short_session_title(str((active or {}).get("title") or "New chat"))
        active_id = str((active or {}).get("id") or "-")
        header = [
            "Session switcher",
            f"Active: {active_title} ({active_id})",
            f"Sessions: {len(sessions)} | Page {normalized_page + 1}/{total_pages}",
        ]
        if notice:
            header.append(f"\n{notice}")
        return "\n".join(header)

    async def _send_sessions_panel(
        self,
        *,
        update: Update | None = None,
        page: int = 0,
        notice: str | None = None,
        query_message: Any | None = None,
    ) -> None:
        if self._session_manager is None:
            if update and update.message:
                await update.message.reply_text("Session panel is unavailable.")
            return

        target_chat_id: str | None = None
        if update and update.effective_chat is not None and update.effective_chat.id is not None:
            target_chat_id = str(update.effective_chat.id)
        elif query_message is not None and getattr(query_message, "chat_id", None) is not None:
            target_chat_id = str(query_message.chat_id)
        if not target_chat_id:
            return

        snapshot = self._session_manager.list_conversation_sessions(
            self._conversation_key(target_chat_id)
        )
        text = self._render_sessions_panel_text(snapshot, page=page, notice=notice)
        keyboard = self._build_sessions_keyboard(snapshot, page=page)

        if query_message is not None:
            try:
                await query_message.edit_text(text=text, reply_markup=keyboard)
            except Exception as e:
                if "Message is not modified" not in str(e):
                    logger.debug(f"Telegram sessions panel edit skipped: {e}")
            return

        if update and update.message:
            await update.message.reply_text(text, reply_markup=keyboard)

    @staticmethod
    def _parse_int(raw: str, default: int = 0) -> int:
        try:
            return int(raw)
        except Exception:
            return default

    async def _on_sessions_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message:
            return
        if not self._is_user_allowed(update.effective_user):
            return
        if self._session_manager is None:
            await self._forward_command(update, context)
            return
        await self._send_sessions_panel(update=update, page=0)

    async def _on_session_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None:
            return

        if not self._is_user_allowed(getattr(query, "from_user", None)):
            try:
                await query.answer(text="You are not allowed to use this bot.", show_alert=True)
            except Exception:
                pass
            return

        try:
            await query.answer()
        except Exception:
            pass

        if self._session_manager is None:
            return

        data = str(query.data or "")
        if not data.startswith(f"{self._SESSION_CB_PREFIX}:"):
            return

        message = query.message
        chat_id_raw = getattr(message, "chat_id", None)
        if chat_id_raw is None:
            return
        conversation_key = self._conversation_key(str(chat_id_raw))

        parts = data.split(":")
        if len(parts) < 2:
            return

        action = parts[1]
        page = 0
        notice: str | None = None

        try:
            if action == "pg":
                page = self._parse_int(parts[2] if len(parts) > 2 else "0")
            elif action == "rf":
                page = self._parse_int(parts[2] if len(parts) > 2 else "0")
            elif action == "new":
                page = self._parse_int(parts[2] if len(parts) > 2 else "0")
                created = self._session_manager.create_session(
                    conversation_key,
                    title=None,
                    switch_to=True,
                )
                notice = f"Started: {created['title']} ({created['id']})"
            elif action == "sw":
                if len(parts) < 3:
                    notice = "Invalid session target."
                else:
                    target = parts[2]
                    page = self._parse_int(parts[3] if len(parts) > 3 else "0")
                    switched = self._session_manager.switch_session(conversation_key, target)
                    notice = f"Switched: {switched['title']} ({switched['id']})"
            elif action in {"pin", "unpin"}:
                if len(parts) < 3:
                    notice = "Invalid session target."
                else:
                    target_raw = parts[2]
                    page = self._parse_int(parts[3] if len(parts) > 3 else "0")
                    target = target_raw
                    if target_raw == "active":
                        current = self._session_manager.list_conversation_sessions(conversation_key)
                        target = str(current.get("active_session_id") or "")
                    if target:
                        updated = self._session_manager.set_session_pinned(
                            conversation_key,
                            target,
                            pinned=action == "pin",
                        )
                        notice = (
                            f"{'Pinned' if action == 'pin' else 'Unpinned'}: "
                            f"{updated['title']} ({updated['id']})"
                        )
            elif action == "helpsearch":
                notice = "Use /session search <keyword> for filtering sessions."
            elif action == "noop":
                pass
            else:
                notice = "Unknown action."
        except KeyError:
            notice = "Session not found. Refreshing list."
        except Exception as e:
            logger.warning(f"Telegram session callback failed: {e}")
            notice = "Session action failed."

        await self._send_sessions_panel(
            page=page,
            notice=notice,
            query_message=message,
        )

    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self.config.token:
            logger.error("Telegram bot token not configured")
            return

        self._running = True

        # Build the application with larger connection pool to avoid pool-timeout on long runs
        req = HTTPXRequest(
            connection_pool_size=16, pool_timeout=5.0, connect_timeout=30.0, read_timeout=30.0
        )
        builder = (
            Application.builder().token(self.config.token).request(req).get_updates_request(req)
        )
        if self.config.proxy:
            builder = builder.proxy(self.config.proxy).get_updates_proxy(self.config.proxy)
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)

        # Add command handlers
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("new", self._forward_command))
        self._app.add_handler(CommandHandler("sessions", self._on_sessions_command))
        self._app.add_handler(CommandHandler("session", self._forward_command))
        self._app.add_handler(CommandHandler("help", self._forward_command))
        self._app.add_handler(CommandHandler("approve", self._forward_command))
        self._app.add_handler(CommandHandler("deny", self._forward_command))
        self._app.add_handler(
            CallbackQueryHandler(
                self._on_session_callback,
                pattern=rf"^{self._SESSION_CB_PREFIX}:",
            )
        )

        # Add message handler for text, photos, voice, documents
        self._app.add_handler(
            MessageHandler(
                (
                    filters.TEXT
                    | filters.PHOTO
                    | filters.VOICE
                    | filters.AUDIO
                    | filters.Document.ALL
                )
                & ~filters.COMMAND,
                self._on_message,
            )
        )

        logger.info("Starting Telegram bot (polling mode)...")

        # Initialize and start polling
        await self._app.initialize()
        await self._app.start()

        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        logger.info(f"Telegram bot @{bot_info.username} connected")

        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            logger.debug("Telegram bot commands registered")
        except Exception as e:
            logger.warning(f"Failed to register bot commands: {e}")

        # Start polling (this runs until stopped)
        updater = self._app.updater
        if updater is not None:
            await updater.start_polling(
                allowed_updates=["message", "callback_query"],
                drop_pending_updates=True,
            )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        # Cancel all typing indicators
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)

        if self._app:
            logger.info("Stopping Telegram bot...")
            updater = self._app.updater
            if updater is not None:
                await updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        if not self._app:
            logger.warning("Telegram bot not running")
            return

        keep_typing = bool(msg.metadata.get("keep_typing")) or bool(
            msg.metadata.get("is_progress_update")
        )
        if not keep_typing:
            self._stop_typing(msg.chat_id)

        try:
            chat_id = int(msg.chat_id)
        except ValueError:
            logger.error(f"Invalid chat_id: {msg.chat_id}")
            return

        text_content = (msg.content or "").strip()
        media_items = [m.strip() for m in msg.media if isinstance(m, str) and m.strip()]

        caption_used = False
        fallback_notes: list[str] = []
        if media_items:
            first_caption = text_content[:1024] if text_content else None
            for idx, media_ref in enumerate(media_items):
                caption = first_caption if idx == 0 and first_caption else None
                sent = await self._send_media(chat_id, media_ref, caption=caption)
                if sent and caption:
                    caption_used = True
                if not sent:
                    fallback_notes.append(f"[media send failed: {media_ref}]")

        if caption_used:
            text_content = text_content[1024:].lstrip()
        if fallback_notes:
            text_content = "\n".join(p for p in [text_content, *fallback_notes] if p)

        if text_content:
            for chunk in _split_message(text_content):
                try:
                    html = _markdown_to_telegram_html(chunk)
                    await self._app.bot.send_message(chat_id=chat_id, text=html, parse_mode="HTML")
                except Exception as e:
                    logger.warning(f"HTML parse failed, falling back to plain text: {e}")
                    try:
                        await self._app.bot.send_message(chat_id=chat_id, text=chunk)
                    except Exception as e2:
                        logger.error(f"Error sending Telegram message: {e2}")

    async def _send_media(self, chat_id: int, media_ref: str, caption: str | None = None) -> bool:
        """Send one media item as photo/document. Supports local path and URL."""
        if not self._app:
            return False

        is_image = self._is_image_ref(media_ref)
        try:
            if media_ref.startswith(("http://", "https://")):
                if is_image:
                    await self._app.bot.send_photo(
                        chat_id=chat_id, photo=media_ref, caption=caption
                    )
                else:
                    await self._app.bot.send_document(
                        chat_id=chat_id, document=media_ref, caption=caption
                    )
                return True

            path = Path(media_ref).expanduser()
            if not path.is_file():
                logger.warning(f"Media file not found: {media_ref}")
                return False

            with path.open("rb") as f:
                if is_image:
                    await self._app.bot.send_photo(chat_id=chat_id, photo=f, caption=caption)
                else:
                    await self._app.bot.send_document(chat_id=chat_id, document=f, caption=caption)
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram media '{media_ref}': {e}")
            return False

    @staticmethod
    def _is_image_ref(media_ref: str) -> bool:
        """Detect whether a media reference looks like an image."""
        lower = media_ref.lower()
        image_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
        if any(lower.endswith(ext) for ext in image_exts):
            return True

        if media_ref.startswith(("http://", "https://")):
            return False

        mime, _ = mimetypes.guess_type(media_ref)
        return bool(mime and mime.startswith("image/"))

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return

        user = update.effective_user
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm nanobot.\n\n"
            "Send me a message and I'll respond!\n"
            "Type /help to see available commands."
        )

    @staticmethod
    def _sender_id(user) -> str:
        """Build sender_id with username for allowlist matching."""
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    def _is_user_allowed(self, user: Any | None) -> bool:
        if user is None:
            return False
        sender_id = self._sender_id(user)
        allowed = self.is_allowed(sender_id)
        if not allowed:
            logger.warning(f"Access denied for telegram user {sender_id} on session controls")
        return allowed

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward slash commands to the bus for unified handling in AgentLoop."""
        if not update.message or not update.effective_user:
            return
        await self._handle_message(
            sender_id=self._sender_id(update.effective_user),
            chat_id=str(update.message.chat_id),
            content=str(update.message.text or ""),
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return

        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        sender_id = self._sender_id(user)

        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id

        # Build content from text and/or media
        content_parts = []
        media_paths = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Handle media files
        media_file = None
        media_type = None

        if message.photo:
            media_file = message.photo[-1]  # Largest photo
            media_type = "image"
        elif message.voice:
            media_file = message.voice
            media_type = "voice"
        elif message.audio:
            media_file = message.audio
            media_type = "audio"
        elif message.document:
            media_file = message.document
            media_type = "file"

        # Download media if present
        if media_file and self._app:
            try:
                file = await self._app.bot.get_file(media_file.file_id)
                ext = self._get_extension(
                    str(media_type or "file"),
                    getattr(media_file, "mime_type", None),
                )

                # Save to workspace/media/
                media_dir = Path.home() / ".nanobot" / "media"
                media_dir.mkdir(parents=True, exist_ok=True)

                file_path = media_dir / f"{media_file.file_id[:16]}{ext}"
                await file.download_to_drive(str(file_path))

                media_paths.append(str(file_path))

                # Handle voice transcription
                if media_type == "voice" or media_type == "audio":
                    from nanobot.providers.transcription import GroqTranscriptionProvider

                    transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                    transcription = await transcriber.transcribe(file_path)
                    if transcription:
                        logger.info(f"Transcribed {media_type}: {transcription[:50]}...")
                        content_parts.append(f"[transcription: {transcription}]")
                    else:
                        content_parts.append(f"[{media_type}: {file_path}]")
                else:
                    content_parts.append(f"[{media_type}: {file_path}]")

                logger.debug(f"Downloaded {media_type} to {file_path}")
            except Exception as e:
                logger.error(f"Failed to download media: {e}")
                content_parts.append(f"[{media_type}: download failed]")

        content = "\n".join(content_parts) if content_parts else "[empty message]"

        logger.debug(f"Telegram message from {sender_id}: {content[:50]}...")

        str_chat_id = str(chat_id)

        # Start typing indicator before processing
        self._start_typing(str_chat_id)

        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str_chat_id,
            content=content,
            media=media_paths,
            metadata={
                "message_id": message.message_id,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_group": message.chat.type != "private",
            },
        )

    def _start_typing(self, chat_id: str) -> None:
        """Start sending 'typing...' indicator for a chat."""
        # Cancel any existing typing task for this chat
        self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def _typing_loop(self, chat_id: str) -> None:
        """Repeatedly send 'typing' action until cancelled."""
        try:
            while self._app:
                await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Typing indicator stopped for {chat_id}: {e}")

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        logger.error(f"Telegram error: {context.error}")

    def _get_extension(self, media_type: str, mime_type: str | None) -> str:
        """Get file extension based on media type."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "audio/ogg": ".ogg",
                "audio/mpeg": ".mp3",
                "audio/mp4": ".m4a",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]

        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "file": ""}
        return type_map.get(media_type, "")
