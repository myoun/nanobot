"""Discord channel implementation using Discord Gateway websocket."""

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx
import websockets
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import DiscordConfig
from nanobot.session.manager import SessionManager


DISCORD_API_BASE = "https://discord.com/api/v10"
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 20MB


class DiscordChannel(BaseChannel):
    """Discord channel using Gateway websocket."""

    name = "discord"
    _SESSION_CB_PREFIX = "nbs"
    _SESSION_PAGE_SIZE = 20
    _SESSION_TITLE_MAX_LEN = 80

    def __init__(
        self,
        config: DiscordConfig,
        bus: MessageBus,
        session_manager: SessionManager | None = None,
    ):
        super().__init__(config, bus)
        self.config: DiscordConfig = config
        self._session_manager = session_manager
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._seq: int | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._http: httpx.AsyncClient | None = None

    @staticmethod
    def _conversation_key(channel_id: str) -> str:
        return f"discord:{channel_id}"

    @classmethod
    def _session_pages(cls, total_items: int) -> int:
        if total_items <= 0:
            return 1
        return (total_items + cls._SESSION_PAGE_SIZE - 1) // cls._SESSION_PAGE_SIZE

    @classmethod
    def _short_session_title(cls, title: str) -> str:
        raw = " ".join((title or "").split()).strip()
        if not raw:
            raw = "New chat"
        if len(raw) <= cls._SESSION_TITLE_MAX_LEN:
            return raw
        return raw[: cls._SESSION_TITLE_MAX_LEN - 1].rstrip() + "..."

    @staticmethod
    def _parse_int(raw: str, default: int = 0) -> int:
        try:
            return int(raw)
        except Exception:
            return default

    def _build_sessions_components(
        self,
        snapshot: dict[str, Any],
        *,
        owner_id: str,
        page: int,
    ) -> list[dict[str, Any]]:
        sessions = snapshot.get("sessions")
        if not isinstance(sessions, list):
            sessions = []

        total_pages = self._session_pages(len(sessions))
        normalized_page = min(max(0, page), total_pages - 1)
        start = normalized_page * self._SESSION_PAGE_SIZE
        end = start + self._SESSION_PAGE_SIZE
        visible = sessions[start:end]

        options: list[dict[str, Any]] = []
        for item in visible:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or "")
            if not sid:
                continue
            options.append(
                {
                    "label": self._short_session_title(str(item.get("title") or "")),
                    "value": sid,
                    "description": sid,
                    "default": bool(item.get("active")),
                }
            )

        if not options:
            options = [
                {
                    "label": "No sessions",
                    "value": "__none__",
                    "description": "Create a session first",
                    "default": True,
                }
            ]

        select_row = {
            "type": 1,
            "components": [
                {
                    "type": 3,
                    "custom_id": (f"{self._SESSION_CB_PREFIX}:sel:{owner_id}:{normalized_page}"),
                    "placeholder": "Select a session",
                    "min_values": 1,
                    "max_values": 1,
                    "options": options,
                    "disabled": options[0]["value"] == "__none__",
                }
            ],
        }

        action_row = {
            "type": 1,
            "components": [
                {
                    "type": 2,
                    "style": 1,
                    "label": "New",
                    "custom_id": f"{self._SESSION_CB_PREFIX}:new:{owner_id}:{normalized_page}",
                },
                {
                    "type": 2,
                    "style": 2,
                    "label": "Refresh",
                    "custom_id": f"{self._SESSION_CB_PREFIX}:rf:{owner_id}:{normalized_page}",
                },
                {
                    "type": 2,
                    "style": 2,
                    "label": "Pin",
                    "custom_id": f"{self._SESSION_CB_PREFIX}:pin:{owner_id}:{normalized_page}",
                },
                {
                    "type": 2,
                    "style": 2,
                    "label": "Unpin",
                    "custom_id": f"{self._SESSION_CB_PREFIX}:unpin:{owner_id}:{normalized_page}",
                },
            ],
        }

        nav_row = {
            "type": 1,
            "components": [
                {
                    "type": 2,
                    "style": 2,
                    "label": "Prev",
                    "custom_id": (
                        f"{self._SESSION_CB_PREFIX}:pg:{owner_id}:{max(0, normalized_page - 1)}"
                    ),
                    "disabled": normalized_page <= 0,
                },
                {
                    "type": 2,
                    "style": 2,
                    "label": "Next",
                    "custom_id": (
                        f"{self._SESSION_CB_PREFIX}:pg:{owner_id}:{min(total_pages - 1, normalized_page + 1)}"
                    ),
                    "disabled": normalized_page >= total_pages - 1,
                },
            ],
        }

        return [select_row, action_row, nav_row]

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

        lines = [
            "Session switcher",
            f"Active: {active_title} ({active_id})",
            f"Sessions: {len(sessions)} | Page {normalized_page + 1}/{total_pages}",
            "Use select/buttons. Pin and unpin apply to active session.",
        ]
        if notice:
            lines.append(f"\n{notice}")
        return "\n".join(lines)

    async def _send_discord_json_message(
        self,
        *,
        channel_id: str,
        payload: dict[str, Any],
    ) -> None:
        if not self._http:
            return
        url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
        headers = {"Authorization": f"Bot {self.config.token}"}
        resp = await self._http.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            logger.warning(f"Discord panel message failed: {resp.status_code} {resp.text[:200]}")

    async def _send_sessions_panel_message(
        self,
        *,
        channel_id: str,
        user_id: str,
        page: int = 0,
        notice: str | None = None,
    ) -> None:
        if self._session_manager is None:
            return
        snapshot = self._session_manager.list_conversation_sessions(
            self._conversation_key(channel_id)
        )
        text = self._render_sessions_panel_text(snapshot, page=page, notice=notice)
        components = self._build_sessions_components(
            snapshot,
            owner_id=user_id,
            page=page,
        )
        await self._send_discord_json_message(
            channel_id=channel_id,
            payload={"content": text, "components": components},
        )

    async def _respond_interaction(
        self,
        *,
        interaction_id: str,
        interaction_token: str,
        response_type: int,
        data: dict[str, Any],
    ) -> None:
        if not self._http:
            return
        url = f"{DISCORD_API_BASE}/interactions/{interaction_id}/{interaction_token}/callback"
        headers = {"Authorization": f"Bot {self.config.token}"}
        resp = await self._http.post(
            url,
            headers=headers,
            json={"type": response_type, "data": data},
        )
        if resp.status_code >= 400:
            logger.warning(
                f"Discord interaction response failed: {resp.status_code} {resp.text[:200]}"
            )

    async def _handle_interaction_create(self, payload: dict[str, Any]) -> None:
        if self._session_manager is None:
            return

        interaction_id = str(payload.get("id") or "")
        interaction_token = str(payload.get("token") or "")
        channel_id = str(payload.get("channel_id") or "")
        if not interaction_id or not interaction_token or not channel_id:
            return

        user_block = (
            ((payload.get("member") or {}).get("user"))
            if isinstance(payload.get("member"), dict)
            else payload.get("user")
        ) or {}
        actor_user_id = str((user_block or {}).get("id") or "")

        data = payload.get("data")
        if not isinstance(data, dict):
            return
        custom_id = str(data.get("custom_id") or "")
        if not custom_id.startswith(f"{self._SESSION_CB_PREFIX}:"):
            return

        parts = custom_id.split(":")
        if len(parts) < 4:
            return
        action = parts[1]
        owner_id = parts[2]
        page = self._parse_int(parts[3], default=0)

        if actor_user_id and owner_id and actor_user_id != owner_id:
            await self._respond_interaction(
                interaction_id=interaction_id,
                interaction_token=interaction_token,
                response_type=4,
                data={
                    "content": "This panel belongs to another user.",
                    "flags": 64,
                },
            )
            return

        conversation_key = self._conversation_key(channel_id)
        notice: str | None = None

        try:
            if action == "sel":
                values = data.get("values")
                if isinstance(values, list) and values:
                    selected = str(values[0])
                    if selected != "__none__":
                        switched = self._session_manager.switch_session(
                            conversation_key,
                            selected,
                        )
                        notice = f"Switched: {switched['title']} ({switched['id']})"
            elif action == "new":
                created = self._session_manager.create_session(
                    conversation_key,
                    title=None,
                    switch_to=True,
                )
                notice = f"Started: {created['title']} ({created['id']})"
            elif action in {"pin", "unpin"}:
                current_snapshot = self._session_manager.list_conversation_sessions(
                    conversation_key
                )
                active_id = str(current_snapshot.get("active_session_id") or "")
                if active_id:
                    updated = self._session_manager.set_session_pinned(
                        conversation_key,
                        active_id,
                        pinned=action == "pin",
                    )
                    notice = (
                        f"{'Pinned' if action == 'pin' else 'Unpinned'}: "
                        f"{updated['title']} ({updated['id']})"
                    )
            elif action == "rf":
                pass
            elif action == "pg":
                pass
            else:
                notice = "Unknown action."
        except KeyError:
            notice = "Session not found. Refreshing list."
        except Exception as e:
            logger.warning(f"Discord session interaction failed: {e}")
            notice = "Session action failed."

        snapshot = self._session_manager.list_conversation_sessions(conversation_key)
        content = self._render_sessions_panel_text(snapshot, page=page, notice=notice)
        components = self._build_sessions_components(snapshot, owner_id=owner_id, page=page)
        await self._respond_interaction(
            interaction_id=interaction_id,
            interaction_token=interaction_token,
            response_type=7,
            data={"content": content, "components": components},
        )

    async def start(self) -> None:
        """Start the Discord gateway connection."""
        if not self.config.token:
            logger.error("Discord bot token not configured")
            return

        self._running = True
        self._http = httpx.AsyncClient(timeout=30.0)

        while self._running:
            try:
                logger.info("Connecting to Discord gateway...")
                async with websockets.connect(self.config.gateway_url) as ws:
                    self._ws = ws
                    await self._gateway_loop()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Discord gateway error: {e}")
                if self._running:
                    logger.info("Reconnecting to Discord gateway in 5 seconds...")
                    await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the Discord channel."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._http:
            await self._http.aclose()
            self._http = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Discord REST API."""
        if not self._http:
            logger.warning("Discord HTTP client not initialized")
            return

        url = f"{DISCORD_API_BASE}/channels/{msg.chat_id}/messages"
        text_content = (msg.content or "").strip()
        media_refs = [m.strip() for m in msg.media if isinstance(m, str) and m.strip()]
        file_media: list[Path] = []
        url_media: list[str] = []
        failed_media: list[str] = []
        skipped_media: list[str] = []

        for ref in media_refs:
            if ref.startswith(("http://", "https://")):
                url_media.append(ref)
                continue
            p = Path(ref).expanduser()
            if p.is_file():
                file_media.append(p)
            else:
                failed_media.append(ref)

        if len(file_media) > 10:
            dropped = file_media[10:]
            file_media = file_media[:10]
            skipped_media.extend(str(p) for p in dropped)

        if url_media:
            text_content = "\n".join(part for part in [text_content, *url_media] if part)
        if failed_media:
            failed_lines = [f"[media not found: {m}]" for m in failed_media]
            text_content = "\n".join(part for part in [text_content, *failed_lines] if part)
        if skipped_media:
            skipped_lines = [
                f"[media skipped: attachment limit exceeded: {m}]" for m in skipped_media
            ]
            text_content = "\n".join(part for part in [text_content, *skipped_lines] if part)

        payload: dict[str, Any] = {}
        if text_content:
            payload["content"] = text_content

        if msg.reply_to:
            payload["message_reference"] = {"message_id": msg.reply_to}
            payload["allowed_mentions"] = {"replied_user": False}

        headers = {"Authorization": f"Bot {self.config.token}"}
        if not payload and not file_media:
            return

        try:
            for attempt in range(3):
                try:
                    response = await self._post_message(url, headers, payload, file_media)
                    if response.status_code == 429:
                        try:
                            data = response.json()
                        except Exception:
                            data = {}
                        retry_after = float(data.get("retry_after", 1.0))
                        logger.warning(f"Discord rate limited, retrying in {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    return
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Error sending Discord message: {e}")
                    else:
                        await asyncio.sleep(1)
        finally:
            await self._stop_typing(msg.chat_id)

    async def _post_message(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        files: list[Path],
    ) -> httpx.Response:
        """Send Discord message using JSON or multipart/form-data for attachments."""
        if not files:
            return await self._http.post(url, headers=headers, json=payload)

        open_files = []
        try:
            multipart_files = []
            for idx, path in enumerate(files):
                f = path.open("rb")
                open_files.append(f)
                mime, _ = mimetypes.guess_type(path.name)
                multipart_files.append(
                    (f"files[{idx}]", (path.name, f, mime or "application/octet-stream"))
                )
            return await self._http.post(
                url,
                headers=headers,
                data={"payload_json": json.dumps(payload)},
                files=multipart_files,
            )
        finally:
            for f in open_files:
                try:
                    f.close()
                except Exception:
                    pass

    async def _gateway_loop(self) -> None:
        """Main gateway loop: identify, heartbeat, dispatch events."""
        if not self._ws:
            return

        async for raw in self._ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from Discord gateway: {raw[:100]}")
                continue

            op = data.get("op")
            event_type = data.get("t")
            seq = data.get("s")
            payload = data.get("d")

            if seq is not None:
                self._seq = seq

            if op == 10:
                # HELLO: start heartbeat and identify
                interval_ms = payload.get("heartbeat_interval", 45000)
                await self._start_heartbeat(interval_ms / 1000)
                await self._identify()
            elif op == 0 and event_type == "READY":
                logger.info("Discord gateway READY")
            elif op == 0 and event_type == "MESSAGE_CREATE":
                await self._handle_message_create(payload)
            elif op == 0 and event_type == "INTERACTION_CREATE":
                await self._handle_interaction_create(payload)
            elif op == 7:
                # RECONNECT: exit loop to reconnect
                logger.info("Discord gateway requested reconnect")
                break
            elif op == 9:
                # INVALID_SESSION: reconnect
                logger.warning("Discord gateway invalid session")
                break

    async def _identify(self) -> None:
        """Send IDENTIFY payload."""
        if not self._ws:
            return

        identify = {
            "op": 2,
            "d": {
                "token": self.config.token,
                "intents": self.config.intents,
                "properties": {
                    "os": "nanobot",
                    "browser": "nanobot",
                    "device": "nanobot",
                },
            },
        }
        await self._ws.send(json.dumps(identify))

    async def _start_heartbeat(self, interval_s: float) -> None:
        """Start or restart the heartbeat loop."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        async def heartbeat_loop() -> None:
            while self._running and self._ws:
                payload = {"op": 1, "d": self._seq}
                try:
                    await self._ws.send(json.dumps(payload))
                except Exception as e:
                    logger.warning(f"Discord heartbeat failed: {e}")
                    break
                await asyncio.sleep(interval_s)

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def _handle_message_create(self, payload: dict[str, Any]) -> None:
        """Handle incoming Discord messages."""
        author = payload.get("author") or {}
        if author.get("bot"):
            return

        sender_id = str(author.get("id", ""))
        channel_id = str(payload.get("channel_id", ""))
        content = payload.get("content") or ""

        if not sender_id or not channel_id:
            return

        if not self.is_allowed(sender_id):
            return

        stripped_content = content.strip()
        if stripped_content == "/sessions" and self._session_manager is not None:
            await self._send_sessions_panel_message(
                channel_id=channel_id,
                user_id=sender_id,
                page=0,
            )
            return

        content_parts = [content] if content else []
        media_paths: list[str] = []
        media_dir = Path.home() / ".nanobot" / "media"

        for attachment in payload.get("attachments") or []:
            url = attachment.get("url")
            filename = attachment.get("filename") or "attachment"
            size = attachment.get("size") or 0
            if not url or not self._http:
                continue
            if size and size > MAX_ATTACHMENT_BYTES:
                content_parts.append(f"[attachment: {filename} - too large]")
                continue
            try:
                media_dir.mkdir(parents=True, exist_ok=True)
                file_path = (
                    media_dir / f"{attachment.get('id', 'file')}_{filename.replace('/', '_')}"
                )
                resp = await self._http.get(url)
                resp.raise_for_status()
                file_path.write_bytes(resp.content)
                media_paths.append(str(file_path))
                content_parts.append(f"[attachment: {file_path}]")
            except Exception as e:
                logger.warning(f"Failed to download Discord attachment: {e}")
                content_parts.append(f"[attachment: {filename} - download failed]")

        reply_to = (payload.get("referenced_message") or {}).get("id")

        await self._start_typing(channel_id)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=channel_id,
            content="\n".join(p for p in content_parts if p) or "[empty message]",
            media=media_paths,
            metadata={
                "message_id": str(payload.get("id", "")),
                "guild_id": payload.get("guild_id"),
                "reply_to": reply_to,
            },
        )

    async def _start_typing(self, channel_id: str) -> None:
        """Start periodic typing indicator for a channel."""
        await self._stop_typing(channel_id)

        async def typing_loop() -> None:
            url = f"{DISCORD_API_BASE}/channels/{channel_id}/typing"
            headers = {"Authorization": f"Bot {self.config.token}"}
            while self._running:
                try:
                    await self._http.post(url, headers=headers)
                except Exception:
                    pass
                await asyncio.sleep(8)

        self._typing_tasks[channel_id] = asyncio.create_task(typing_loop())

    async def _stop_typing(self, channel_id: str) -> None:
        """Stop typing indicator for a channel."""
        task = self._typing_tasks.pop(channel_id, None)
        if task:
            task.cancel()
