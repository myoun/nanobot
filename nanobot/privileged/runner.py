"""Minimal privileged runner (Unix socket server, allowlist actions only)."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import grp
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from nanobot.utils.helpers import ensure_dir


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_packages(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    packages: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return []
        pkg = item.strip()
        if not pkg:
            return []
        if not all(ch.isalnum() or ch in "+.-" for ch in pkg):
            return []
        packages.append(pkg)
    return packages


class PrivilegedRunner:
    """Allowlist-only privileged command runner."""

    def __init__(self, socket_path: str, audit_log_path: str, socket_group: str | None = None):
        self.socket_path = Path(socket_path)
        self.audit_log_path = Path(audit_log_path)
        self.socket_group = socket_group
        ensure_dir(self.audit_log_path.parent)

    async def serve(self) -> None:
        if os.name != "posix":
            raise RuntimeError("Privileged runner is supported only on Unix/Linux.")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()

        server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        self._apply_socket_permissions()
        async with server:
            await server.serve_forever()

    def _apply_socket_permissions(self) -> None:
        if self.socket_group:
            try:
                gid = grp.getgrnam(self.socket_group).gr_gid
            except KeyError as e:
                raise RuntimeError(f"Socket group not found: {self.socket_group}") from e
            try:
                os.chown(self.socket_path, -1, gid)
            except PermissionError as e:
                raise RuntimeError(
                    f"Failed to set socket group '{self.socket_group}' on {self.socket_path}: {e}"
                ) from e
        os.chmod(self.socket_path, 0o660)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        result: dict[str, Any]
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=15)
            if not raw:
                result = {"ok": False, "error": "Empty request"}
            else:
                payload = json.loads(raw.decode("utf-8", "replace"))
                result = await asyncio.to_thread(self._execute_payload, payload)
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        writer.write((json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    def _execute_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or "")
        action = str(payload.get("action") or "")
        action_args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        timeout_s = int(payload.get("timeout_s") or 300)
        timeout_s = max(10, min(timeout_s, 900))

        if action == "shell_command":
            cmd = str(action_args.get("command") or "").strip()
            if not cmd:
                result = {
                    "ok": False,
                    "request_id": request_id,
                    "action": action,
                    "error": "Invalid command for shell_command",
                }
                self._audit(result)
                return result

            completed = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                shell=True,
                executable="/bin/bash",
                check=False,
            )
            result = {
                "ok": completed.returncode == 0,
                "request_id": request_id,
                "action": action,
                "exit_code": completed.returncode,
                "stdout": (completed.stdout or "").strip(),
                "stderr": (completed.stderr or "").strip(),
            }
            if completed.returncode != 0:
                result["error"] = "Privileged command failed"
            self._audit(result)
            return result

        try:
            commands = self._resolve_commands(action, action_args)
        except ValueError as e:
            result = {
                "ok": False,
                "request_id": request_id,
                "action": action,
                "error": str(e),
            }
            self._audit(result)
            return result

        combined_stdout: list[str] = []
        combined_stderr: list[str] = []
        for argv in commands:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                shell=False,
                check=False,
            )
            if completed.stdout:
                combined_stdout.append(completed.stdout.strip())
            if completed.stderr:
                combined_stderr.append(completed.stderr.strip())
            if completed.returncode != 0:
                result = {
                    "ok": False,
                    "request_id": request_id,
                    "action": action,
                    "exit_code": completed.returncode,
                    "stdout": "\n".join(s for s in combined_stdout if s),
                    "stderr": "\n".join(s for s in combined_stderr if s),
                    "error": "Privileged command failed",
                }
                self._audit(result)
                return result

        result = {
            "ok": True,
            "request_id": request_id,
            "action": action,
            "stdout": "\n".join(s for s in combined_stdout if s),
            "stderr": "\n".join(s for s in combined_stderr if s),
            "exit_code": 0,
        }
        self._audit(result)
        return result

    @staticmethod
    def _resolve_commands(action: str, args: dict[str, Any]) -> list[list[str]]:
        if action == "apt_update":
            return [["/usr/bin/apt-get", "update"]]

        if action == "apt_install":
            packages = _safe_packages(args.get("packages"))
            if not packages:
                raise ValueError("Invalid packages for apt_install")
            return [["/usr/bin/apt-get", "install", "-y", *packages]]

        if action == "apt_remove":
            packages = _safe_packages(args.get("packages"))
            if not packages:
                raise ValueError("Invalid packages for apt_remove")
            return [["/usr/bin/apt-get", "remove", "-y", *packages]]

        if action == "apt_autoremove":
            return [["/usr/bin/apt-get", "autoremove", "-y"]]

        if action == "apt_update_install":
            packages = _safe_packages(args.get("packages"))
            if not packages:
                raise ValueError("Invalid packages for apt_update_install")
            return [
                ["/usr/bin/apt-get", "update"],
                ["/usr/bin/apt-get", "install", "-y", *packages],
            ]

        if action == "apt_remove_autoremove":
            packages = _safe_packages(args.get("packages"))
            if not packages:
                raise ValueError("Invalid packages for apt_remove_autoremove")
            return [
                ["/usr/bin/apt-get", "remove", "-y", *packages],
                ["/usr/bin/apt-get", "autoremove", "-y"],
            ]

        raise ValueError(f"Unsupported privileged action: {action}")

    def _audit(self, result: dict[str, Any]) -> None:
        entry = {"ts": _now_iso(), **result}
        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="nanobot privileged runner")
    parser.add_argument("--socket", default="/run/nanobot-privileged.sock", help="Unix socket path")
    parser.add_argument(
        "--audit-log",
        default="/var/log/nanobot-privileged.log",
        help="Audit log JSONL path",
    )
    parser.add_argument(
        "--socket-group",
        default="",
        help="Socket group for non-root client access (e.g., your login group).",
    )
    args = parser.parse_args()

    runner = PrivilegedRunner(
        socket_path=args.socket,
        audit_log_path=args.audit_log,
        socket_group=(args.socket_group.strip() or None),
    )
    asyncio.run(runner.serve())


if __name__ == "__main__":
    main()
