"""LangSmith tracing helper with graceful fallback when dependency is absent."""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from typing import Any

from loguru import logger

try:
    from langsmith.run_trees import RunTree as _RunTree
except ImportError:
    _RunTree = None

_TRUE_VALUES = {"1", "true", "yes", "on"}
_RUN_STACK: ContextVar[tuple[Any, ...]] = ContextVar("langsmith_run_stack", default=())


def _env_enabled() -> bool:
    return (
        os.getenv("LANGSMITH_TRACING", "").strip().lower() in _TRUE_VALUES
        or os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in _TRUE_VALUES
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            sanitized[str(key)] = _json_safe(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


class _NoopSpan:
    def set_outputs(self, outputs: Any) -> None:
        return

    def add_metadata(self, metadata: dict[str, Any] | None) -> None:
        return

    def finish(self, error: BaseException | None = None) -> None:
        return


class _LiveSpan:
    def __init__(self, run: Any, token: Token[tuple[Any, ...]]):
        self._run = run
        self._token = token
        self._outputs: Any = None
        self._extra_metadata: dict[str, Any] = {}
        self._finished = False

    def set_outputs(self, outputs: Any) -> None:
        self._outputs = _json_safe(outputs)

    def add_metadata(self, metadata: dict[str, Any] | None) -> None:
        if isinstance(metadata, dict):
            self._extra_metadata.update(_json_safe(metadata))

    def finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        try:
            if self._extra_metadata:
                extra = getattr(self._run, "extra", None)
                if not isinstance(extra, dict):
                    extra = {}
                existing = extra.get("metadata")
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(self._extra_metadata)
                extra["metadata"] = existing
                self._run.extra = extra
            if error is None:
                outputs = self._outputs if self._outputs is not None else {}
                self._run.end(outputs=outputs)
            else:
                self._run.end(error=str(error))
            self._run.patch()
        except Exception as exc:
            logger.debug(f"LangSmith span finalize failed: {exc}")
        finally:
            try:
                _RUN_STACK.reset(self._token)
            except Exception:
                pass
            self._finished = True


class LangSmithTracer:
    """Create parent/child LangSmith runs without requiring LangGraph."""

    def __init__(self) -> None:
        self.enabled = _env_enabled() and _RunTree is not None
        if _env_enabled() and _RunTree is None:
            logger.warning(
                "LANGSMITH_TRACING is enabled but langsmith is not installed; tracing disabled."
            )

    def start_span(
        self,
        name: str,
        *,
        run_type: str = "chain",
        inputs: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> _NoopSpan | _LiveSpan:
        if not self.enabled or _RunTree is None:
            return _NoopSpan()

        payload_inputs = _json_safe(inputs) if inputs is not None else {}
        payload_metadata = _json_safe(metadata) if metadata is not None else {}
        payload_tags = [str(tag) for tag in tags] if isinstance(tags, list) else None
        kwargs: dict[str, Any] = {
            "name": name,
            "run_type": run_type,
            "inputs": payload_inputs,
        }
        if payload_metadata:
            kwargs["extra"] = {"metadata": payload_metadata}
        if payload_tags:
            kwargs["tags"] = payload_tags

        current_stack = _RUN_STACK.get()
        parent = current_stack[-1] if current_stack else None

        try:
            if parent is not None and hasattr(parent, "create_child"):
                run = parent.create_child(**kwargs)
            else:
                run = _RunTree(**kwargs)
            run.post()
        except Exception as exc:
            logger.debug(f"LangSmith span creation failed ({name}): {exc}")
            return _NoopSpan()

        token = _RUN_STACK.set(current_stack + (run,))
        return _LiveSpan(run, token)


_TRACER: LangSmithTracer | None = None


def get_langsmith_tracer() -> LangSmithTracer:
    global _TRACER
    if _TRACER is None:
        _TRACER = LangSmithTracer()
    return _TRACER
