from __future__ import annotations

import json
import secrets
import string
from collections.abc import Mapping
from typing import Any

KEY_TYPE = "type"
KEY_TEXT = "text"
KEY_SID = "sid"
KEY_TS = "ts"
KEY_CODE = "code"

TYPE_HELLO = "hello"
TYPE_USER_MESSAGE = "user_message"
TYPE_ASSISTANT_MESSAGE = "assistant_message"
TYPE_PROGRESS_UPDATE = "progress_update"
TYPE_ERROR = "error"

ERR_BAD_JSON = "bad_json"
ERR_BAD_MESSAGE = "bad_message"
ERR_BUSY = "busy"
ERR_UNAUTHORIZED = "unauthorized"
ERR_RATE_LIMIT = "rate_limit"

CLOSE_AUTH_FAILED = 4401
CLOSE_PROTOCOL_ERROR = 4400

_SESSION_ALPHABET = string.ascii_letters + string.digits


def new_session_id(length: int = 16) -> str:
    safe_len = max(6, min(64, length))
    return "".join(secrets.choice(_SESSION_ALPHABET) for _ in range(safe_len))


def safe_json_loads(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except Exception:
        return None

    if not isinstance(payload, Mapping):
        return None

    return dict(payload)


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()
