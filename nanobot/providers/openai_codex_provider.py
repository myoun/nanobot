"""OpenAI Codex Responses Provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, AsyncGenerator

import httpx
from loguru import logger

from oauth_cli_kit import get_token as get_codex_token
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "nanobot"
BUILTIN_WEB_SEARCH_TOOL = {"type": "web_search"}


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    def __init__(self, default_model: str = "openai-codex/gpt-5.1-codex"):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        model = model or self.default_model
        system_prompt, input_items = _convert_messages(messages)

        token = await asyncio.to_thread(get_codex_token)
        headers = _build_headers(token.account_id, token.access)

        body: dict[str, Any] = {
            "model": _strip_model_prefix(model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": [
                "reasoning.encrypted_content",
                "web_search_call.action.sources",
            ],
            "prompt_cache_key": _prompt_cache_key(messages),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        # Always expose Codex built-in web search and filter out legacy function
        # tools named "web_search" to avoid ambiguous tool selection.
        body["tools"] = _build_response_tools(tools)

        url = DEFAULT_CODEX_URL

        try:
            try:
                content, tool_calls, finish_reason, metadata = await _request_codex(url, headers, body, verify=True)
            except Exception as e:
                if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                    raise
                logger.warning("SSL certificate verification failed for Codex API; retrying with verify=False")
                content, tool_calls, finish_reason, metadata = await _request_codex(url, headers, body, verify=False)
            response = LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                metadata=metadata,
            )
            self._log_response_debug(response, model=model)
            return response
        except Exception as e:
            error_response = LLMResponse(
                content=f"Error calling Codex: {str(e)}",
                finish_reason="error",
            )
            self._log_response_debug(error_response, model=model)
            return error_response

    def get_default_model(self) -> str:
        return self.default_model


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/"):
        return model.split("/", 1)[1]
    return model


def _build_headers(account_id: str, token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "nanobot (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool,
) -> tuple[str, list[ToolCallRequest], str, dict[str, Any]]:
    async with httpx.AsyncClient(timeout=60.0, verify=verify) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(_friendly_error(response.status_code, text.decode("utf-8", "ignore")))
            return await _consume_sse(response)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling schema to Codex flat format."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params if isinstance(params, dict) else {},
        })
    return converted


def _build_response_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Build Responses API tools with built-in web search enabled."""
    converted = _convert_tools(tools or [])
    filtered = [
        tool for tool in converted
        if not (tool.get("type") == "function" and tool.get("name") == "web_search")
    ]
    filtered.append(BUILTIN_WEB_SEARCH_TOOL.copy())
    return filtered


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_prompt = ""
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            # Handle text first.
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                        "status": "completed",
                        "id": f"msg_{idx}",
                    }
                )
            # Then handle tool calls.
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                call_id = call_id or f"call_{idx}"
                item_id = item_id or f"fc_{idx}"
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            output_text = content if isinstance(content, str) else json.dumps(content)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
            continue

    return system_prompt, input_items


def _convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def _prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                parsed = _parse_sse_buffer(buffer)
                buffer = []
                if parsed is not None:
                    yield parsed
            continue
        buffer.append(line)
    # Some servers terminate the stream without a trailing blank line.
    if buffer:
        parsed = _parse_sse_buffer(buffer)
        if parsed is not None:
            yield parsed


async def _consume_sse(response: httpx.Response) -> tuple[str, list[ToolCallRequest], str, dict[str, Any]]:
    text_segments: dict[str, str] = {}
    text_segment_order: list[str] = []
    message_text_fallback_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    web_search_trace: list[dict[str, Any]] = []
    seen_web_actions: set[str] = set()
    finish_reason = "stop"

    async for event in _iter_sse(response):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                tool_call_buffers[call_id] = {
                    "id": item.get("id") or "fc_0",
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "",
                }
        elif event_type == "response.output_text.delta":
            _append_text_segment(
                text_segments,
                text_segment_order,
                _text_segment_key(event),
                event.get("delta") or "",
            )
        elif event_type == "response.output_text.done":
            _merge_text_segment(
                text_segments,
                text_segment_order,
                _text_segment_key(event),
                _output_text_done_value(event),
            )
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"raw": args_raw}
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                        name=buf.get("name") or item.get("name"),
                        arguments=args,
                    )
                )
            elif item.get("type") == "web_search_call":
                _append_web_search_action(
                    web_search_trace,
                    seen_web_actions,
                    _normalize_web_search_action(item.get("action")),
                )
            elif item.get("type") == "message":
                text = _extract_message_output_text(item)
                if text:
                    message_text_fallback_parts.append(text)
        elif event_type.startswith("response.web_search_call."):
            web_item = event.get("web_search_call") or {}
            _append_web_search_action(
                web_search_trace,
                seen_web_actions,
                _normalize_web_search_action(web_item.get("action") or event.get("action")),
            )
        elif event_type == "response.completed":
            status = (event.get("response") or {}).get("status")
            finish_reason = _map_finish_reason(status)
        elif event_type in {"error", "response.failed"}:
            raise RuntimeError("Codex response failed")

    content = "".join(text_segments[key] for key in text_segment_order if text_segments.get(key))
    if not content and message_text_fallback_parts:
        # Fallback for variants that only emit final message items.
        content = "".join(message_text_fallback_parts)

    metadata: dict[str, Any] = {}
    if web_search_trace:
        metadata["web_search_trace"] = web_search_trace
    return content, tool_calls, finish_reason, metadata


_FINISH_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "error", "cancelled": "error"}


def _parse_sse_buffer(buffer: list[str]) -> dict[str, Any] | None:
    data_lines = [line[5:].strip() for line in buffer if line.startswith("data:")]
    if not data_lines:
        return None
    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _text_segment_key(event: dict[str, Any]) -> str:
    item_id = event.get("item_id")
    output_index = event.get("output_index")
    content_index = event.get("content_index")
    if any(value is not None for value in (item_id, output_index, content_index)):
        return f"{item_id or 'item'}:{output_index or 0}:{content_index or 0}"
    return "global"


def _append_text_segment(
    segments: dict[str, str],
    order: list[str],
    key: str,
    text: str,
) -> None:
    if not text:
        return
    if key not in segments:
        segments[key] = ""
        order.append(key)
    segments[key] += text


def _merge_text_segment(
    segments: dict[str, str],
    order: list[str],
    key: str,
    text: str,
) -> None:
    if not text:
        return
    if key not in segments:
        segments[key] = text
        order.append(key)
        return
    current = segments[key]
    if not current:
        segments[key] = text
        return
    if text.startswith(current):
        segments[key] = text
        return
    if current.startswith(text) or current.endswith(text):
        return
    if text.endswith(current):
        segments[key] = text
        return
    segments[key] = current + text


def _output_text_done_value(event: dict[str, Any]) -> str:
    text = event.get("text")
    if isinstance(text, str):
        return text
    output_text = event.get("output_text")
    if isinstance(output_text, str):
        return output_text
    if isinstance(output_text, dict):
        nested_text = output_text.get("text")
        if isinstance(nested_text, str):
            return nested_text
    return ""


def _extract_message_output_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "output_text":
            text = entry.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def _map_finish_reason(status: str | None) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def _append_web_search_action(
    trace: list[dict[str, Any]],
    seen: set[str],
    action: dict[str, Any] | None,
) -> None:
    if not action:
        return
    key = json.dumps(action, ensure_ascii=True, sort_keys=True)
    if key in seen:
        return
    seen.add(key)
    trace.append(action)


def _normalize_web_search_action(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action_type = raw.get("type")
    if not isinstance(action_type, str) or not action_type:
        return None

    normalized: dict[str, Any] = {"type": action_type}

    query = raw.get("query")
    if isinstance(query, str) and query:
        normalized["query"] = query

    queries = raw.get("queries")
    if isinstance(queries, list):
        cleaned_queries = [q for q in queries if isinstance(q, str) and q]
        if cleaned_queries:
            normalized["queries"] = cleaned_queries

    url = raw.get("url")
    if isinstance(url, str) and url:
        normalized["url"] = url

    pattern = raw.get("pattern")
    if isinstance(pattern, str) and pattern:
        normalized["pattern"] = pattern

    sources = raw.get("sources")
    if isinstance(sources, list):
        cleaned_sources = [s for s in sources if isinstance(s, dict)]
        if cleaned_sources:
            normalized["sources"] = cleaned_sources

    return normalized


def _friendly_error(status_code: int, raw: str) -> str:
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    return f"HTTP {status_code}: {raw}"
