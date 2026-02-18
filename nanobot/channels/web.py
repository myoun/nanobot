from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from loguru import logger
from websockets.legacy.server import serve as ws_serve

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.web_protocol import (
    CLOSE_AUTH_FAILED,
    CLOSE_PROTOCOL_ERROR,
    ERR_BAD_JSON,
    ERR_BAD_MESSAGE,
    ERR_BUSY,
    ERR_RATE_LIMIT,
    ERR_UNAUTHORIZED,
    KEY_ACTION,
    KEY_CODE,
    KEY_HISTORY,
    KEY_SESSIONS,
    KEY_SESSION_ID,
    KEY_TITLE,
    KEY_SID,
    KEY_TEXT,
    KEY_TS,
    KEY_TYPE,
    TYPE_ASSISTANT_MESSAGE,
    TYPE_ERROR,
    TYPE_HELLO,
    TYPE_PROGRESS_UPDATE,
    TYPE_SESSION_ACTION,
    TYPE_SESSIONS_STATE,
    TYPE_USER_MESSAGE,
    coerce_text,
    new_session_id,
    safe_json_loads,
)
from nanobot.config.schema import WebConfig
from nanobot.session.manager import SessionManager


class WebChannel(BaseChannel):
    name = "web"

    def __init__(
        self,
        config: WebConfig,
        bus: MessageBus,
        session_manager: SessionManager | None = None,
    ):
        super().__init__(config, bus)
        self.config: WebConfig = config
        self._session_manager = session_manager
        self._server: Any = None
        self.bound_port: int = int(config.port)
        self._connections: dict[str, set[Any]] = defaultdict(set)
        self._pending: set[str] = set()
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._index_html = self._load_index_html()

    async def start(self) -> None:
        self._validate_security_settings()
        self._server = await cast(Any, ws_serve)(
            self._ws_handler,
            self.config.host,
            self.config.port,
            process_request=self._process_request,
            origins=self.config.allow_origins or None,
            max_size=self.config.max_message_bytes,
            max_queue=self.config.max_queue_frames,
            ping_interval=self.config.ping_interval_s,
            ping_timeout=self.config.ping_timeout_s,
        )
        sockets = getattr(self._server, "sockets", None) or []
        if sockets:
            self.bound_port = int(sockets[0].getsockname()[1])
        self._running = True
        logger.info(f"Web channel listening on {self.config.host}:{self.bound_port}")

    async def stop(self) -> None:
        self._running = False
        for ws in self._all_connections():
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()
        self._pending.clear()
        self._recent.clear()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("Web channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        sid = coerce_text(msg.chat_id)
        if not sid:
            return

        is_progress = bool((msg.metadata or {}).get("is_progress_update"))
        payload_type = TYPE_PROGRESS_UPDATE if is_progress else TYPE_ASSISTANT_MESSAGE
        payload = {
            KEY_TYPE: payload_type,
            KEY_TEXT: msg.content,
            KEY_SID: sid,
            KEY_TS: int(time.time()),
        }

        targets = list(self._connections.get(sid, set()))
        dead: list[Any] = []
        for ws in targets:
            try:
                await ws.send(json.dumps(payload, ensure_ascii=True))
            except Exception:
                dead.append(ws)

        if dead:
            for ws in dead:
                self._connections[sid].discard(ws)

        session_state = self._extract_session_state(msg.metadata)
        if session_state:
            await self._broadcast_sessions_state(sid, session_state)

        if payload_type == TYPE_ASSISTANT_MESSAGE:
            self._pending.discard(sid)

    def _load_index_html(self) -> str:
        index_path = Path(__file__).resolve().parent.parent / "web" / "index.html"
        try:
            return index_path.read_text(encoding="utf-8")
        except OSError:
            return "<html><body><h1>nanobot web</h1></body></html>"

    def _validate_security_settings(self) -> None:
        host = self.config.host.strip().lower()
        is_local = host in {"127.0.0.1", "localhost", "::1"}
        if not is_local:
            if not self.config.token:
                raise ValueError("channels.web.token is required for non-local host binding")
            if not self.config.allow_origins:
                raise ValueError(
                    "channels.web.allow_origins is required for non-local host binding"
                )

    async def _process_request(self, path: str, request_headers: Any) -> Any:
        parsed = urlparse(path)
        if parsed.path == "/":
            return self._http_response(
                HTTPStatus.OK, self._index_html, content_type="text/html; charset=utf-8"
            )
        if parsed.path == "/health":
            return self._http_response(
                HTTPStatus.OK, "ok\n", content_type="text/plain; charset=utf-8"
            )
        if parsed.path == "/ws":
            return None
        return self._http_response(
            HTTPStatus.NOT_FOUND, "not found\n", content_type="text/plain; charset=utf-8"
        )

    def _http_response(
        self, status: HTTPStatus, body: str, *, content_type: str
    ) -> tuple[HTTPStatus, list[tuple[str, str]], bytes]:
        payload = body.encode("utf-8")
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(payload))),
            ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff"),
            (
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self' ws: wss:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
            ),
        ]
        return status, headers, payload

    async def _ws_handler(self, websocket: Any) -> None:
        path = coerce_text(getattr(websocket, "path", ""))
        parsed = urlparse(path)
        if parsed.path != "/ws":
            await websocket.close(code=CLOSE_PROTOCOL_ERROR, reason="invalid websocket path")
            return

        query = parse_qs(parsed.query)
        sid = coerce_text((query.get("sid") or [""])[0]) or new_session_id()
        token = coerce_text((query.get("token") or [""])[0])

        if self.config.token and token != self.config.token:
            await self._send_error(websocket, sid, ERR_UNAUTHORIZED, "unauthorized")
            await websocket.close(code=CLOSE_AUTH_FAILED, reason="unauthorized")
            return

        self._connections[sid].add(websocket)
        await websocket.send(
            json.dumps(
                {
                    KEY_TYPE: TYPE_HELLO,
                    KEY_SID: sid,
                    KEY_TEXT: "connected",
                    KEY_TS: int(time.time()),
                },
                ensure_ascii=True,
            )
        )

        try:
            async for raw in websocket:
                data = safe_json_loads(raw)
                if data is None:
                    await self._send_error(websocket, sid, ERR_BAD_JSON, "invalid json payload")
                    continue

                msg_type = coerce_text(data.get(KEY_TYPE))
                if msg_type == TYPE_SESSION_ACTION:
                    await self._handle_session_action(websocket, sid, data)
                    continue

                text = coerce_text(data.get(KEY_TEXT))
                if msg_type != TYPE_USER_MESSAGE or not text:
                    await self._send_error(
                        websocket,
                        sid,
                        ERR_BAD_MESSAGE,
                        "expected user_message with non-empty text",
                    )
                    continue

                if sid in self._pending:
                    await self._send_error(websocket, sid, ERR_BUSY, "request already in progress")
                    continue

                if not self._allow_rate(sid):
                    await self._send_error(websocket, sid, ERR_RATE_LIMIT, "rate limit exceeded")
                    continue

                self._pending.add(sid)
                selected_session_id = coerce_text(data.get(KEY_SESSION_ID))
                metadata: dict[str, Any] = {"web": {"sid": sid}}
                if selected_session_id:
                    metadata["web"]["session_id"] = selected_session_id
                    metadata["session"] = {"id": selected_session_id}
                await self._handle_message(
                    sender_id=sid,
                    chat_id=sid,
                    content=text,
                    metadata=metadata,
                )
        except Exception as e:
            logger.warning(f"WebSocket session ended for sid={sid}: {e}")
        finally:
            self._connections[sid].discard(websocket)
            if not self._connections[sid]:
                self._connections.pop(sid, None)

    async def _send_error(self, websocket: Any, sid: str, code: str, text: str) -> None:
        payload = {
            KEY_TYPE: TYPE_ERROR,
            KEY_CODE: code,
            KEY_TEXT: text,
            KEY_SID: sid,
            KEY_TS: int(time.time()),
        }
        await websocket.send(json.dumps(payload, ensure_ascii=True))

    @staticmethod
    def _extract_session_state(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(metadata, dict):
            return None
        nanobot_meta = metadata.get("_nanobot")
        if not isinstance(nanobot_meta, dict):
            return None
        state = nanobot_meta.get("session_state")
        return state if isinstance(state, dict) else None

    async def _broadcast_sessions_state(self, sid: str, snapshot: dict[str, Any]) -> None:
        snapshot = self._normalize_sessions_snapshot(sid, snapshot)
        payload = {
            KEY_TYPE: TYPE_SESSIONS_STATE,
            KEY_SID: sid,
            KEY_TS: int(time.time()),
            "active_session_id": snapshot.get("active_session_id"),
            KEY_SESSIONS: snapshot.get("sessions", []),
            KEY_HISTORY: snapshot.get("history", []),
        }
        dead: list[Any] = []
        for ws in list(self._connections.get(sid, set())):
            try:
                await ws.send(json.dumps(payload, ensure_ascii=True))
            except Exception:
                dead.append(ws)

        if dead:
            for ws in dead:
                self._connections[sid].discard(ws)

    def _normalize_sessions_snapshot(
        self, sid: str, snapshot: dict[str, Any] | None
    ) -> dict[str, Any]:
        current = dict(snapshot or {})
        if self._session_manager is None:
            if not isinstance(current.get("history"), list):
                current["history"] = []
            return current

        conversation_key = self._conversation_key_for_sid(sid)
        current = self._session_manager.list_conversation_sessions(conversation_key)
        current["history"] = self._active_session_history(current)
        return current

    def _active_session_history(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        if self._session_manager is None:
            return []

        active_id = str(snapshot.get("active_session_id") or "")
        if not active_id:
            return []

        active_key = ""
        for item in snapshot.get("sessions", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("id") or "") == active_id:
                active_key = str(item.get("key") or "")
                break
        if not active_key:
            return []

        session = self._session_manager.get_or_create(active_key)
        history: list[dict[str, Any]] = []
        for item in session.messages:
            role = coerce_text(item.get("role"))
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content") or "")
            if not content:
                continue
            msg = {"role": role, "content": content}
            if timestamp := item.get("timestamp"):
                msg["timestamp"] = str(timestamp)
            history.append(msg)
        return history

    def _conversation_key_for_sid(self, sid: str) -> str:
        return f"{self.name}:{sid}"

    async def _handle_session_action(self, websocket: Any, sid: str, data: dict[str, Any]) -> None:
        if self._session_manager is None:
            await self._send_error(websocket, sid, ERR_BAD_MESSAGE, "session manager unavailable")
            return

        action = coerce_text(data.get(KEY_ACTION)).lower()
        conversation_key = self._conversation_key_for_sid(sid)

        try:
            if action in {"", "list"}:
                snapshot = self._session_manager.list_conversation_sessions(conversation_key)
                await self._broadcast_sessions_state(sid, snapshot)
                return

            if action in {"new", "create"}:
                raw_title = coerce_text(data.get(KEY_TITLE))
                self._session_manager.create_session(
                    conversation_key,
                    title=raw_title or None,
                    switch_to=True,
                )
                snapshot = self._session_manager.list_conversation_sessions(conversation_key)
                await self._broadcast_sessions_state(sid, snapshot)
                return

            if action == "switch":
                target_id = coerce_text(data.get(KEY_SESSION_ID))
                if not target_id:
                    await self._send_error(
                        websocket,
                        sid,
                        ERR_BAD_MESSAGE,
                        "session_id is required for switch",
                    )
                    return
                self._session_manager.switch_session(conversation_key, target_id)
                snapshot = self._session_manager.list_conversation_sessions(conversation_key)
                await self._broadcast_sessions_state(sid, snapshot)
                return

            if action == "rename":
                target_id = coerce_text(data.get(KEY_SESSION_ID))
                title = coerce_text(data.get(KEY_TITLE))
                if not target_id or not title:
                    await self._send_error(
                        websocket,
                        sid,
                        ERR_BAD_MESSAGE,
                        "session_id and title are required for rename",
                    )
                    return
                self._session_manager.rename_session(
                    conversation_key,
                    target_id,
                    title,
                    auto_title=False,
                )
                snapshot = self._session_manager.list_conversation_sessions(conversation_key)
                await self._broadcast_sessions_state(sid, snapshot)
                return

            if action == "delete":
                target_id = coerce_text(data.get(KEY_SESSION_ID))
                if not target_id:
                    await self._send_error(
                        websocket,
                        sid,
                        ERR_BAD_MESSAGE,
                        "session_id is required for delete",
                    )
                    return
                self._session_manager.delete_session(conversation_key, target_id)
                snapshot = self._session_manager.list_conversation_sessions(conversation_key)
                await self._broadcast_sessions_state(sid, snapshot)
                return
        except KeyError:
            await self._send_error(websocket, sid, ERR_BAD_MESSAGE, "unknown session")
            return
        except Exception as e:
            logger.warning(f"Web session action failed for sid={sid}: {e}")
            await self._send_error(websocket, sid, ERR_BAD_MESSAGE, "session action failed")
            return

        await self._send_error(websocket, sid, ERR_BAD_MESSAGE, "unknown session action")

    def _allow_rate(self, sid: str) -> bool:
        now = time.monotonic()
        window = float(self.config.rate_limit_window_s)
        limit = int(self.config.rate_limit_count)
        q = self._recent[sid]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True

    def _all_connections(self) -> list[Any]:
        out: list[Any] = []
        for conns in self._connections.values():
            out.extend(conns)
        return out
