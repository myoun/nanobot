"""OpenAI Codex provider backed by Codex App Server for primary turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import AppServerTurnResult
from nanobot.providers.codex_app_server_client import CodexAppServerClient
from nanobot.providers.openai_codex_provider import OpenAICodexProvider, _strip_model_prefix


class OpenAICodexAppServerProvider(OpenAICodexProvider):
    """Use direct Codex Responses for helper calls and App Server for main turns."""

    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.3-codex",
        *,
        workspace: Path | None = None,
        app_server_client: CodexAppServerClient | None = None,
        profile_name: str | None = None,
        use_workspace_profile: bool = True,
        sandbox: str = "danger-full-access",
    ):
        super().__init__(default_model=default_model)
        self.workspace = str((workspace or Path.cwd()).resolve())
        self.sandbox = sandbox
        self._app_server = app_server_client or CodexAppServerClient(
            cwd=self.workspace,
            client_name="nanobot",
            client_title="nanobot",
            client_version="0",
            profile_name=profile_name,
            use_workspace_profile=use_workspace_profile,
        )

    @property
    def uses_app_server(self) -> bool:
        return True

    @property
    def supports_native_web_search(self) -> bool:
        return True

    async def run_app_server_turn(
        self,
        *,
        thread_id: str | None,
        input_items: list[dict[str, Any]],
        tools: ToolRegistry,
        developer_instructions: str,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        cwd: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        exclude_tool_names: Iterable[str] | None = None,
    ) -> AppServerTurnResult:
        """Run one App Server turn with nanobot dynamic tools."""
        active_model = _strip_model_prefix(model or self.default_model)
        excluded = {name for name in (exclude_tool_names or []) if name}
        dynamic_tools = [
            spec
            for spec in tools.get_dynamic_tool_specs()
            if str(spec.get("name") or "") not in excluded
        ]
        resolved_thread_id = await self._app_server.ensure_thread(
            thread_id=thread_id,
            dynamic_tools=dynamic_tools,
            developer_instructions=developer_instructions,
            cwd=str(Path(cwd or self.workspace).resolve()),
            model=active_model,
            sandbox=self.sandbox,
        )
        turn_id, final_text, tools_used, metadata = await self._app_server.run_turn(
            thread_id=resolved_thread_id,
            input_items=input_items,
            tool_executor=tools.execute_dynamic,
            event_callback=event_callback,
            cwd=str(Path(cwd or self.workspace).resolve()),
            model=active_model,
            effort=reasoning_effort,
        )
        return AppServerTurnResult(
            thread_id=resolved_thread_id,
            turn_id=turn_id,
            final_text=final_text,
            tools_used=tools_used,
            metadata=metadata,
        )

    async def get_runtime_status(self) -> dict[str, Any]:
        """Return App Server runtime/account status for user-facing diagnostics."""
        return await self._app_server.get_runtime_status()

    async def aclose(self) -> None:
        await self._app_server.aclose()
