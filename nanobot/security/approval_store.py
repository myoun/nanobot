"""Persistent store for privileged execution approvals."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import uuid

from nanobot.utils.helpers import ensure_dir


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


@dataclass
class ApprovalRequest:
    request_id: str
    session_key: str
    channel: str
    chat_id: str
    requester_id: str
    command: str
    working_dir: str
    action: str
    action_args: dict
    status: str
    created_at: str
    expires_at: str
    resolved_at: str | None = None
    resolver_id: str | None = None
    result_preview: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"


class ApprovalStore:
    """Simple JSON-backed store for approval-gated privileged requests."""

    def __init__(self, path: Path, ttl_seconds: int = 600, single_pending_per_chat: bool = True):
        self.path = path
        self.ttl_seconds = max(30, ttl_seconds)
        self.single_pending_per_chat = single_pending_per_chat
        ensure_dir(path.parent)

    def create_pending(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        requester_id: str,
        command: str,
        working_dir: str,
        action: str,
        action_args: dict,
    ) -> ApprovalRequest:
        data = self._load_data()
        self._expire_pending(data)
        if self.single_pending_per_chat:
            pending = self._find_pending(data, session_key)
            if pending:
                return pending

        now = _now_utc()
        req = ApprovalRequest(
            request_id=f"apr_{uuid.uuid4().hex[:10]}",
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            requester_id=requester_id,
            command=command,
            working_dir=working_dir,
            action=action,
            action_args=action_args,
            status="pending",
            created_at=_iso(now) or "",
            expires_at=_iso(now + timedelta(seconds=self.ttl_seconds)) or "",
        )
        data["requests"].append(asdict(req))
        self._save_data(data)
        return req

    def get_pending(self, session_key: str) -> ApprovalRequest | None:
        data = self._load_data()
        changed = self._expire_pending(data)
        req = self._find_pending(data, session_key)
        if changed:
            self._save_data(data)
        return req

    def resolve(
        self,
        session_key: str,
        *,
        status: str,
        resolver_id: str,
        result_preview: str | None = None,
    ) -> ApprovalRequest | None:
        data = self._load_data()
        self._expire_pending(data)
        req = self._find_pending(data, session_key)
        if not req:
            self._save_data(data)
            return None

        for item in data["requests"]:
            if item.get("request_id") == req.request_id:
                item["status"] = status
                item["resolved_at"] = _iso(_now_utc())
                item["resolver_id"] = resolver_id
                item["result_preview"] = result_preview
                break

        self._save_data(data)
        return self._find_by_id(data, req.request_id)

    def _find_pending(self, data: dict, session_key: str) -> ApprovalRequest | None:
        for item in reversed(data["requests"]):
            if item.get("session_key") != session_key:
                continue
            if item.get("status") == "pending":
                return self._to_request(item)
        return None

    def _find_by_id(self, data: dict, request_id: str) -> ApprovalRequest | None:
        for item in data["requests"]:
            if item.get("request_id") == request_id:
                return self._to_request(item)
        return None

    @staticmethod
    def _to_request(item: dict) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=str(item.get("request_id", "")),
            session_key=str(item.get("session_key", "")),
            channel=str(item.get("channel", "")),
            chat_id=str(item.get("chat_id", "")),
            requester_id=str(item.get("requester_id", "")),
            command=str(item.get("command", "")),
            working_dir=str(item.get("working_dir", "")),
            action=str(item.get("action", "")),
            action_args=item.get("action_args", {}) if isinstance(item.get("action_args"), dict) else {},
            status=str(item.get("status", "")),
            created_at=str(item.get("created_at", "")),
            expires_at=str(item.get("expires_at", "")),
            resolved_at=item.get("resolved_at"),
            resolver_id=item.get("resolver_id"),
            result_preview=item.get("result_preview"),
        )

    def _expire_pending(self, data: dict) -> bool:
        now = _now_utc()
        changed = False
        for item in data["requests"]:
            if item.get("status") != "pending":
                continue
            expires = _from_iso(item.get("expires_at"))
            if expires and expires < now:
                item["status"] = "expired"
                item["resolved_at"] = _iso(now)
                changed = True
        return changed

    def _load_data(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "requests": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {"version": 1, "requests": []}
            requests = raw.get("requests")
            if not isinstance(requests, list):
                requests = []
            return {"version": 1, "requests": requests}
        except Exception:
            return {"version": 1, "requests": []}

    def _save_data(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

