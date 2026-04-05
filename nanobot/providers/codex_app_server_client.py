"""Codex App Server stdio client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
from typing import Any, Awaitable, Callable, Sequence

from loguru import logger

from nanobot.providers.codex_profile import CodexProfileManager

DynamicToolExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
AppServerEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass
class _TurnAccumulator:
    """Collect notifications for a single in-flight App Server turn."""

    thread_id: str
    turn_id: str | None = None
    done: asyncio.Future[tuple[str, str, list[str], dict[str, Any]]] | None = None
    _agent_message_text: dict[str, str] = field(default_factory=dict)
    _agent_message_order: list[str] = field(default_factory=list)
    _agent_message_phase: dict[str, str] = field(default_factory=dict)
    _tools_used: list[str] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)

    def set_turn_id(self, turn_id: str | None) -> None:
        if turn_id:
            self.turn_id = turn_id

    def _remember_agent_item(self, item_id: str) -> None:
        if item_id not in self._agent_message_order:
            self._agent_message_order.append(item_id)
        self._agent_message_text.setdefault(item_id, "")

    def handle_notification(self, message: dict[str, Any]) -> bool:
        """Process a notification. Returns True when the turn is complete."""
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return False

        thread_id = params.get("threadId")
        if thread_id != self.thread_id:
            return False

        if method == "turn/started":
            turn = params.get("turn") or {}
            if isinstance(turn, dict):
                self.set_turn_id(str(turn.get("id") or ""))
            return False

        if self.turn_id and params.get("turnId") not in {None, self.turn_id}:
            return False

        if method == "item/agentMessage/delta":
            item_id = str(params.get("itemId") or "")
            delta = str(params.get("delta") or "")
            if item_id:
                self._remember_agent_item(item_id)
                self._agent_message_text[item_id] += delta
            return False

        if method in {"item/started", "item/completed"}:
            item = params.get("item") or {}
            if not isinstance(item, dict):
                return False
            item_type = str(item.get("type") or "")
            if item_type == "dynamicToolCall":
                tool = str(item.get("tool") or "")
                if tool:
                    self._tools_used.append(tool)
            elif item_type == "agentMessage":
                item_id = str(item.get("id") or "")
                if item_id:
                    self._remember_agent_item(item_id)
                    phase = str(item.get("phase") or "").strip()
                    if phase:
                        self._agent_message_phase[item_id] = phase
                    text = item.get("text")
                    if isinstance(text, str):
                        self._agent_message_text[item_id] = text
            return False

        if method == "thread/tokenUsage/updated":
            self._metadata["token_usage"] = params.get("tokenUsage")
            return False

        if method == "turn/completed":
            turn = params.get("turn") or {}
            turn_id = str((turn or {}).get("id") or "")
            if turn_id:
                self.set_turn_id(turn_id)
            final_text = ""
            if self._agent_message_order:
                last_id = self._agent_message_order[-1]
                final_text = self._agent_message_text.get(last_id, "")
            if self.done and not self.done.done():
                self.done.set_result((
                    self.turn_id or turn_id,
                    final_text,
                    list(self._tools_used),
                    dict(self._metadata),
                ))
            return True

        if method == "error":
            error = params.get("message") or message.get("error") or "Unknown App Server error"
            if self.done and not self.done.done():
                self.done.set_exception(RuntimeError(str(error)))
            return True

        return False

    def phase_for(self, item_id: str) -> str:
        return str(self._agent_message_phase.get(item_id) or "")


class CodexAppServerClient:
    """Minimal JSON-RPC client for `codex app-server --listen stdio://`."""

    _STDIO_LIMIT = 4 * 1024 * 1024

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        client_name: str = "nanobot",
        client_title: str = "nanobot",
        client_version: str = "0",
        profile_name: str | None = None,
        use_workspace_profile: bool = True,
    ):
        self.cwd = str(Path(cwd).resolve()) if cwd else None
        self.use_workspace_profile = use_workspace_profile
        self.profile_name = profile_name or os.environ.get("NANOBOT_CODEX_PROFILE_NAME", "nanobot").strip() or "nanobot"
        self.command = list(
            command
            or self._default_command(
                profile_name=self.profile_name if self.use_workspace_profile else None
            )
        )
        self.client_name = client_name
        self.client_title = client_title
        self.client_version = client_version
        self._profile_manager = (
            CodexProfileManager(Path(self.cwd), profile_name=self.profile_name)
            if self.cwd and self.use_workspace_profile
            else None
        )

        self._proc: asyncio.subprocess.Process | None = None
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._request_seq = 0
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._loaded_threads: set[str] = set()
        self._active_turns: dict[str, _TurnAccumulator] = {}
        self._tool_executors: dict[str, DynamicToolExecutor] = {}
        self._event_callbacks: dict[str, AppServerEventCallback] = {}
        self._account_info: dict[str, Any] = {}
        self._config_info: dict[str, Any] = {}
        self._rate_limit_snapshot: dict[str, Any] = {}
        self._initialized = False
        self._closing = False

    @staticmethod
    def _default_command(profile_name: str | None = None) -> list[str]:
        env_command = os.environ.get("NANOBOT_CODEX_APP_SERVER_COMMAND", "").strip()
        if env_command:
            return CodexAppServerClient._inject_profile(shlex.split(env_command), profile_name)

        if codex_path := shutil.which("codex"):
            return CodexAppServerClient._inject_profile(
                [codex_path, "app-server", "--listen", "stdio://"],
                profile_name,
            )

        codex_home = os.environ.get("CODEX_HOME", "").strip()
        if codex_home:
            env_fallback = Path(codex_home).expanduser() / "bin" / "wsl" / "codex"
            if env_fallback.exists():
                return CodexAppServerClient._inject_profile(
                    [str(env_fallback), "app-server", "--listen", "stdio://"],
                    profile_name,
                )

        fallback = Path.home() / ".codex" / "bin" / "wsl" / "codex"
        if fallback.exists():
            return CodexAppServerClient._inject_profile(
                [str(fallback), "app-server", "--listen", "stdio://"],
                profile_name,
            )

        return CodexAppServerClient._inject_profile(
            ["codex", "app-server", "--listen", "stdio://"],
            profile_name,
        )

    @staticmethod
    def _inject_profile(command: list[str], profile_name: str | None) -> list[str]:
        if not command or not profile_name:
            return command
        if "--profile" in command:
            return command
        executable = Path(command[0]).name.lower()
        if "codex" not in executable:
            return command
        return [command[0], "--profile", profile_name, *command[1:]]

    async def ensure_started(self) -> None:
        """Start the App Server subprocess and initialize the session."""
        if self._initialized and self._proc and self._proc.returncode is None:
            return

        async with self._start_lock:
            if self._initialized and self._proc and self._proc.returncode is None:
                return

            if self._profile_manager is not None:
                self._profile_manager.ensure_profile()

            self._proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=self._STDIO_LIMIT,
                cwd=self.cwd,
            )
            self._reader_task = asyncio.create_task(self._reader_loop())
            self._stderr_task = asyncio.create_task(self._stderr_loop())
            self._closing = False

            await self._send_request(
                "initialize",
                {
                    "clientInfo": {
                        "name": self.client_name,
                        "title": self.client_title,
                        "version": self.client_version,
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self._send_notification("initialized")
            self._initialized = True

    async def ensure_thread(
        self,
        *,
        thread_id: str | None,
        dynamic_tools: list[dict[str, Any]],
        developer_instructions: str,
        cwd: str,
        model: str | None = None,
        approval_policy: str = "never",
        sandbox: str = "workspace-write",
    ) -> str:
        """Start or resume an App Server thread with the given tool set."""
        await self.ensure_started()

        params: dict[str, Any] = {
            "model": model,
            "cwd": cwd,
            "approvalPolicy": approval_policy,
            "sandbox": sandbox,
            "developerInstructions": developer_instructions,
            "dynamicTools": dynamic_tools,
            "experimentalRawEvents": False,
            "persistExtendedHistory": False,
        }

        if thread_id:
            if thread_id in self._loaded_threads:
                return thread_id
            result = await self._send_request(
                "thread/resume",
                {
                    "threadId": thread_id,
                    "cwd": cwd,
                    "model": model,
                    "approvalPolicy": approval_policy,
                    "sandbox": sandbox,
                    "developerInstructions": developer_instructions,
                    "dynamicTools": dynamic_tools,
                    "persistExtendedHistory": False,
                },
            )
        else:
            result = await self._send_request("thread/start", params)

        resolved_thread_id = str(((result or {}).get("thread") or {}).get("id") or "")
        if not resolved_thread_id:
            raise RuntimeError("Codex App Server did not return a thread id")
        self._loaded_threads.add(resolved_thread_id)
        return resolved_thread_id

    async def run_turn(
        self,
        *,
        thread_id: str,
        input_items: list[dict[str, Any]],
        tool_executor: DynamicToolExecutor,
        event_callback: AppServerEventCallback | None = None,
        cwd: str,
        model: str | None = None,
        effort: str | None = None,
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        """Run one App Server turn and return turn_id, final_text, tools_used, metadata."""
        await self.ensure_started()

        if thread_id in self._active_turns:
            raise RuntimeError(f"App Server thread {thread_id} already has an active turn")

        loop = asyncio.get_running_loop()
        accumulator = _TurnAccumulator(
            thread_id=thread_id,
            done=loop.create_future(),
        )
        self._active_turns[thread_id] = accumulator
        self._tool_executors[thread_id] = tool_executor
        if event_callback is not None:
            self._event_callbacks[thread_id] = event_callback
        try:
            result = await self._send_request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": input_items,
                    "cwd": cwd,
                    "model": model,
                    "effort": effort,
                },
            )
            turn = (result or {}).get("turn") or {}
            if isinstance(turn, dict):
                accumulator.set_turn_id(str(turn.get("id") or ""))
            turn_id, final_text, tools_used, metadata = await accumulator.done
            return turn_id, final_text, tools_used, metadata
        finally:
            self._active_turns.pop(thread_id, None)
            self._tool_executors.pop(thread_id, None)
            self._event_callbacks.pop(thread_id, None)

    async def get_runtime_status(self) -> dict[str, Any]:
        """Return best-effort account and rate-limit status from App Server."""
        await self.ensure_started()

        try:
            result = await self._send_request("account/read", {"refreshToken": False})
            self._update_account_info(result if isinstance(result, dict) else {})
        except Exception as exc:
            logger.debug("Failed to read Codex account status: {}", exc)

        try:
            params: dict[str, Any] = {"includeLayers": False}
            if self.cwd:
                params["cwd"] = self.cwd
            result = await self._send_request("config/read", params)
            self._update_config_info(result if isinstance(result, dict) else {})
        except Exception as exc:
            logger.debug("Failed to read Codex runtime config: {}", exc)

        try:
            result = await self._send_request("account/rateLimits/read")
            self._update_rate_limit_snapshot(result if isinstance(result, dict) else {})
        except Exception as exc:
            logger.debug("Failed to read Codex rate limits: {}", exc)

        return {
            "account": dict(self._account_info),
            "config": dict(self._config_info),
            "rate_limits": dict(self._rate_limit_snapshot),
        }

    async def aclose(self) -> None:
        """Terminate the App Server process and fail pending requests."""
        proc = self._proc
        self._proc = None
        self._initialized = False
        self._closing = True
        self._loaded_threads.clear()
        self._account_info.clear()
        self._config_info.clear()
        self._rate_limit_snapshot.clear()
        self._fail_pending(RuntimeError("Codex App Server client closed"))

        if proc and proc.stdin:
            try:
                proc.stdin.close()
                wait_closed = getattr(proc.stdin, "wait_closed", None)
                if callable(wait_closed):
                    await wait_closed()
            except (BrokenPipeError, ConnectionResetError, ProcessLookupError, RuntimeError, ValueError):
                pass
            except Exception:
                pass

        if proc and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
        if proc:
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass

        tasks = [task for task in (self._reader_task, self._stderr_task) if task]
        self._reader_task = None
        self._stderr_task = None
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        await self._write_json(payload)

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("Codex App Server process is not running")

        self._request_seq += 1
        request_id = str(self._request_seq)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        payload: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        await self._write_json(payload)
        return await future

    async def _write_json(self, payload: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("Codex App Server process is not running")
        async with self._write_lock:
            self._proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await self._proc.stdin.drain()

    async def _reader_loop(self) -> None:
        if not self._proc or not self._proc.stdout:
            return

        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("Codex App Server closed its stdout stream")
                try:
                    message = json.loads(line.decode("utf-8"))
                except Exception as exc:
                    logger.warning("Failed to decode Codex App Server message: {}", exc)
                    continue
                await self._handle_incoming(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                self._fail_pending(exc)
                logger.warning("Codex App Server reader stopped: {}", exc)

    async def _stderr_loop(self) -> None:
        if not self._proc or not self._proc.stderr:
            return

        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", "ignore").rstrip()
                if text:
                    logger.debug("codex app-server stderr: {}", text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Codex App Server stderr reader stopped: {}", exc)

    async def _handle_incoming(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            await self._handle_server_request(message)
            return

        if "id" in message:
            request_id = str(message.get("id") or "")
            future = self._pending.pop(request_id, None)
            if not future:
                return
            error = message.get("error")
            if error is not None:
                future.set_exception(RuntimeError(str(error)))
            else:
                result = message.get("result")
                if isinstance(result, dict):
                    thread = result.get("thread")
                    if isinstance(thread, dict):
                        thread_id = str(thread.get("id") or "")
                        if thread_id:
                            self._loaded_threads.add(thread_id)
                future.set_result(result)
            return

        method = str(message.get("method") or "")
        params = message.get("params") or {}
        if method == "thread/started":
            thread = params.get("thread") if isinstance(params, dict) else None
            if isinstance(thread, dict):
                thread_id = str(thread.get("id") or "")
                if thread_id:
                    self._loaded_threads.add(thread_id)

        if method == "account/updated" and isinstance(params, dict):
            self._update_account_info(params)

        if method == "account/rateLimits/updated" and isinstance(params, dict):
            self._update_rate_limit_snapshot(params)

        if method == "thread/closed" and isinstance(params, dict):
            thread_id = str(params.get("threadId") or "")
            if thread_id:
                self._loaded_threads.discard(thread_id)

        thread_id = ""
        if isinstance(params, dict):
            thread_id = str(
                params.get("threadId")
                or ((params.get("thread") or {}).get("id") if isinstance(params.get("thread"), dict) else "")
                or ""
            )
        if thread_id and thread_id in self._active_turns:
            accumulator = self._active_turns[thread_id]
            event = self._normalize_turn_event(message, accumulator)
            if event is not None:
                await self._emit_thread_event(thread_id, event)
            accumulator.handle_notification(message)

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        try:
            if method == "item/tool/call":
                thread_id = str(params.get("threadId") or "")
                tool_name = str(params.get("tool") or "")
                arguments = params.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                logger.info(
                    "Codex App Server tool call: {}({})",
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False)[:200],
                )
                await self._emit_thread_event(
                    thread_id,
                    {
                        "type": "tool_call",
                        "thread_id": thread_id,
                        "turn_id": str(params.get("turnId") or ""),
                        "call_id": str(params.get("callId") or ""),
                        "tool": tool_name,
                        "arguments": arguments,
                    },
                )
                executor = self._tool_executors.get(thread_id)
                if executor is None:
                    result = {
                        "contentItems": [{"type": "inputText", "text": f"Error: no tool executor for thread {thread_id}"}],
                        "success": False,
                    }
                else:
                    result = await executor(tool_name, arguments)
            elif method == "account/chatgptAuthTokens/refresh":
                from oauth_cli_kit import get_token as get_codex_token

                token = await asyncio.to_thread(get_codex_token)
                plan_type = getattr(token, "plan_type", None)
                result = {
                    "accessToken": token.access,
                    "chatgptAccountId": token.account_id,
                    "chatgptPlanType": plan_type if isinstance(plan_type, str) else None,
                }
            elif method == "item/commandExecution/requestApproval":
                result = {"decision": "decline"}
            elif method == "item/fileChange/requestApproval":
                result = {"decision": "decline"}
            elif method == "item/permissions/requestApproval":
                result = {"permissions": {}, "scope": "turn"}
            elif method == "item/tool/requestUserInput":
                result = {"answers": {}}
            elif method == "mcpServer/elicitation/request":
                result = {"action": "decline", "content": None, "_meta": None}
            elif method in {"applyPatchApproval", "execCommandApproval"}:
                result = {"decision": "denied"}
            else:
                raise RuntimeError(f"Unsupported Codex App Server request: {method}")

            await self._write_json({"id": request_id, "result": result})
        except Exception as exc:
            logger.warning("Failed to resolve App Server request {}: {}", method, exc)
            await self._write_json(
                {
                    "id": request_id,
                    "error": {"code": -32000, "message": str(exc)},
                }
            )

    async def _emit_thread_event(self, thread_id: str, event: dict[str, Any]) -> None:
        callback = self._event_callbacks.get(thread_id)
        if callback is None:
            return
        try:
            result = callback(event)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.debug("Codex App Server event callback failed: {}", exc)

    @staticmethod
    def _normalize_turn_event(
        message: dict[str, Any],
        accumulator: _TurnAccumulator,
    ) -> dict[str, Any] | None:
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return None

        if method == "item/agentMessage/delta":
            item_id = str(params.get("itemId") or "")
            delta = str(params.get("delta") or "")
            return {
                "type": "agent_delta",
                "thread_id": accumulator.thread_id,
                "turn_id": str(params.get("turnId") or accumulator.turn_id or ""),
                "item_id": item_id,
                "phase": accumulator.phase_for(item_id),
                "delta": delta,
            }

        if method == "item/completed":
            item = params.get("item") or {}
            if isinstance(item, dict) and str(item.get("type") or "") == "dynamicToolCall":
                content_items = item.get("contentItems")
                if not isinstance(content_items, list):
                    content_items = []
                return {
                    "type": "tool_result",
                    "thread_id": accumulator.thread_id,
                    "turn_id": str(params.get("turnId") or accumulator.turn_id or ""),
                    "call_id": str(item.get("id") or ""),
                    "tool": str(item.get("tool") or ""),
                    "success": bool(item.get("success")),
                    "content_items": content_items,
                    "result_preview": CodexAppServerClient._content_items_preview(content_items),
                }

        if method == "thread/tokenUsage/updated":
            return {
                "type": "token_usage",
                "thread_id": accumulator.thread_id,
                "turn_id": str(params.get("turnId") or accumulator.turn_id or ""),
                "token_usage": params.get("tokenUsage"),
            }

        return None

    @staticmethod
    def _content_items_preview(content_items: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in content_items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "inputText":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                continue
            if item_type == "inputImage":
                parts.append("[image content]")
        return "\n\n".join(parts).strip()

    def _fail_pending(self, exc: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()
        for accumulator in self._active_turns.values():
            if accumulator.done and not accumulator.done.done():
                accumulator.done.set_exception(exc)

    def _update_account_info(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        account = payload.get("account")
        if isinstance(account, dict):
            self._account_info["type"] = str(account.get("type") or "").strip() or None
            self._account_info["email"] = str(account.get("email") or "").strip() or None
            self._account_info["planType"] = str(account.get("planType") or "").strip() or None
        auth_mode = str(payload.get("authMode") or "").strip()
        if auth_mode:
            self._account_info["authMode"] = auth_mode
        requires_auth = payload.get("requiresOpenaiAuth")
        if isinstance(requires_auth, bool):
            self._account_info["requiresOpenaiAuth"] = requires_auth
        if "authMode" not in self._account_info:
            account_type = str(self._account_info.get("type") or "").strip()
            if account_type == "chatgpt":
                self._account_info["authMode"] = "chatgpt"
            elif account_type == "apiKey":
                self._account_info["authMode"] = "apikey"

    def _update_rate_limit_snapshot(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        snapshot = self._select_rate_limit_snapshot(payload)
        if snapshot:
            self._rate_limit_snapshot = snapshot

    def _update_config_info(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        config = payload.get("config")
        if not isinstance(config, dict):
            return
        for key in ("model", "model_context_window", "model_auto_compact_token_limit"):
            value = config.get(key)
            if value is not None:
                self._config_info[key] = value

    @staticmethod
    def _select_rate_limit_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        by_limit_id = payload.get("rateLimitsByLimitId")
        if isinstance(by_limit_id, dict):
            preferred = by_limit_id.get("codex")
            if isinstance(preferred, dict):
                return dict(preferred)
            for value in by_limit_id.values():
                if isinstance(value, dict):
                    return dict(value)
        snapshot = payload.get("rateLimits")
        if isinstance(snapshot, dict):
            return dict(snapshot)
        return {}
