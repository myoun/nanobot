"""Client for approval-gated privileged runner."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any


class PrivilegedClient:
    """Unix socket client for the privileged runner."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path

    async def execute(
        self,
        *,
        request_id: str,
        action: str,
        action_args: dict[str, Any],
        timeout_s: int = 300,
    ) -> dict[str, Any]:
        if os.name != "posix":
            return {
                "ok": False,
                "error": "Privileged runner is supported only on Unix/Linux.",
            }

        socket = Path(self.socket_path)
        if not socket.exists():
            return {
                "ok": False,
                "error": f"Privileged runner socket not found: {self.socket_path}",
            }

        payload = {
            "request_id": request_id,
            "action": action,
            "args": action_args,
            "timeout_s": max(10, timeout_s),
        }

        try:
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
            writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), timeout=max(15, timeout_s + 5))
            writer.close()
            await writer.wait_closed()
            if not raw:
                return {"ok": False, "error": "Privileged runner returned empty response"}
            try:
                parsed = json.loads(raw.decode("utf-8", "replace"))
                if isinstance(parsed, dict):
                    return parsed
                return {"ok": False, "error": "Invalid response payload from privileged runner"}
            except Exception:
                return {"ok": False, "error": "Failed to parse privileged runner response"}
        except Exception as e:
            return {"ok": False, "error": f"Failed to contact privileged runner: {e}"}
