"""CLI commands for nanobot."""

import asyncio
import json
import os
import signal
from pathlib import Path
import select
import sys
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from nanobot.bus.events import OutboundMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from nanobot import __version__, __logo__
from nanobot.config.schema import Config
from nanobot.cron.targeting import resolve_delivery_target
from nanobot.utils.helpers import get_workspace_path, sync_workspace_templates

app = typer.Typer(
    name="nanobot",
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios

        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios

        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    history_file = Path.home() / ".nanobot" / "history" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,  # Enter submits (single line mode)
    )


def _extract_web_search_trace(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []
    nanobot_meta = metadata.get("_nanobot")
    if not isinstance(nanobot_meta, dict):
        return []
    trace = nanobot_meta.get("web_search_trace")
    if not isinstance(trace, list):
        return []
    return [item for item in trace if isinstance(item, dict)]


def _render_web_search_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        return

    console.print("[dim]web search trace[/dim]")
    for idx, item in enumerate(trace, start=1):
        action_type = item.get("type")
        if action_type == "search":
            query = item.get("query")
            if not isinstance(query, str) or not query:
                queries = item.get("queries")
                if isinstance(queries, list):
                    query = " | ".join(q for q in queries if isinstance(q, str) and q)
            label = f"search: {query}" if query else "search"
        elif action_type == "open_page":
            url = item.get("url")
            label = f"open_page: {url}" if isinstance(url, str) and url else "open_page"
        elif action_type == "find_in_page":
            url = item.get("url")
            pattern = item.get("pattern")
            if isinstance(url, str) and url and isinstance(pattern, str) and pattern:
                label = f"find_in_page: {pattern} @ {url}"
            elif isinstance(url, str) and url:
                label = f"find_in_page: {url}"
            else:
                label = "find_in_page"
        else:
            label = action_type if isinstance(action_type, str) else "web_action"
        console.print(f"[dim]{idx}. {label}[/dim]")


def _print_agent_response(
    response: str,
    render_markdown: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    web_trace = _extract_web_search_trace(metadata)
    console.print()
    console.print(f"[cyan]{__logo__} nanobot[/cyan]")
    _render_web_search_trace(web_trace)
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


def _contains_config_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value:
            return True
        return any(_contains_config_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_config_key(item, key) for item in value)
    return False


def _warn_deprecated_config_fields(config_path: Path | None) -> None:
    if not config_path or not config_path.exists():
        return
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if _contains_config_key(raw, "memoryWindow"):
        print("Config field `memoryWindow` is no longer used and will be ignored.")


def _load_runtime_config(
    *,
    workspace: str | None = None,
    config: str | None = None,
) -> tuple[Config, Path | None]:
    from nanobot.config.loader import load_config, set_config_path

    config_path = Path(config).expanduser().resolve() if config else None
    if config_path is not None:
        set_config_path(config_path)
    loaded = load_config(config_path)
    _warn_deprecated_config_fields(config_path)
    if workspace:
        loaded.agents.defaults.workspace = str(Path(workspace).expanduser().resolve())
    return loaded, config_path


def _cron_store_path() -> Path:
    from nanobot.config.paths import get_cron_dir

    return get_cron_dir() / "jobs.json"


def _migrate_workspace_cron_store(config: Config) -> None:
    """Merge legacy workspace cron data into the canonical instance store."""
    workspace_store = config.workspace_path / "cron" / "jobs.json"
    target_store = _cron_store_path()
    if not workspace_store.exists():
        return

    if not target_store.exists():
        target_store.parent.mkdir(parents=True, exist_ok=True)
        workspace_store.replace(target_store)
        return

    try:
        workspace_doc = json.loads(workspace_store.read_text(encoding="utf-8"))
        target_doc = json.loads(target_store.read_text(encoding="utf-8"))
    except Exception:
        return

    workspace_jobs = workspace_doc.get("jobs")
    target_jobs = target_doc.get("jobs")
    if not isinstance(workspace_jobs, list) or not isinstance(target_jobs, list):
        return

    merged_jobs: list[dict[str, Any]] = []
    merged_index: dict[str, int] = {}
    for raw_job in target_jobs:
        if not isinstance(raw_job, dict):
            continue
        job_id = raw_job.get("id")
        if not isinstance(job_id, str) or not job_id:
            continue
        merged_index[job_id] = len(merged_jobs)
        merged_jobs.append(raw_job)

    for raw_job in workspace_jobs:
        if not isinstance(raw_job, dict):
            continue
        job_id = raw_job.get("id")
        if not isinstance(job_id, str) or not job_id:
            continue
        existing_idx = merged_index.get(job_id)
        if existing_idx is None:
            merged_index[job_id] = len(merged_jobs)
            merged_jobs.append(raw_job)
            continue
        existing = merged_jobs[existing_idx]
        existing_updated = existing.get("updatedAtMs")
        incoming_updated = raw_job.get("updatedAtMs")
        if not isinstance(existing_updated, int):
            existing_updated = -1
        if not isinstance(incoming_updated, int):
            incoming_updated = -1
        if incoming_updated >= existing_updated:
            merged_jobs[existing_idx] = raw_job

    version = target_doc.get("version", workspace_doc.get("version", 1))
    if not isinstance(version, int):
        version = 1
    target_store.write_text(
        json.dumps({"version": version, "jobs": merged_jobs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    workspace_store.unlink(missing_ok=True)


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True),
):
    """nanobot - Personal AI Assistant."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    wizard: bool = typer.Option(False, "--wizard", help="Use interactive wizard"),
):
    """Initialize nanobot configuration and workspace."""
    from nanobot.config.loader import get_config_path, load_config, save_config, set_config_path
    from nanobot.config.schema import Config

    if config:
        config_path = Path(config).expanduser().resolve()
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")
    else:
        config_path = get_config_path()

    def _apply_workspace_override(loaded: Config) -> Config:
        if workspace:
            loaded.agents.defaults.workspace = str(Path(workspace).expanduser().resolve())
        return loaded

    if config_path.exists():
        if wizard:
            loaded = load_config(config_path)
            runtime_config = _apply_workspace_override(loaded)
        else:
            console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
            console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
            console.print(
                "  [bold]N[/bold] = refresh config, keeping existing values and adding new fields"
            )
            if typer.confirm("Overwrite?"):
                runtime_config = _apply_workspace_override(Config())
                save_config(runtime_config, config_path)
                console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
            else:
                runtime_config = _apply_workspace_override(load_config(config_path))
                save_config(runtime_config, config_path)
                console.print(
                    f"[green]✓[/green] Config refreshed at {config_path} "
                    "(existing values preserved)"
                )
    else:
        runtime_config = _apply_workspace_override(Config())
        if not wizard:
            save_config(runtime_config, config_path)
            console.print(f"[green]✓[/green] Created config at {config_path}")

    if wizard:
        from nanobot.cli.onboard import run_onboard

        try:
            result = run_onboard(initial_config=runtime_config)
        except Exception as exc:
            console.print(f"[red]✗[/red] Error during configuration: {exc}")
            console.print("[yellow]Please run 'nanobot onboard' again to complete setup.[/yellow]")
            raise typer.Exit(1) from exc

        if not result.should_save:
            console.print("[yellow]Configuration discarded. No changes were saved.[/yellow]")
            return

        runtime_config = result.config
        save_config(runtime_config, config_path)
        console.print(f"[green]✓[/green] Config saved at {config_path}")

    _onboard_plugins(config_path)

    workspace_path = get_workspace_path(runtime_config.workspace_path)
    if not workspace_path.exists():
        workspace_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace_path}")

    sync_workspace_templates(workspace_path)

    codex_config_changed = _configure_codex_profile_on_onboard(
        runtime_config,
        workspace_path,
        prompt_if_unset=(not wizard and sys.stdin.isatty() and sys.stdout.isatty()),
    )
    if codex_config_changed:
        save_config(runtime_config, config_path)

    agent_cmd = 'nanobot agent -m "Hello!"'
    gateway_cmd = "nanobot gateway"
    if config:
        agent_cmd += f" --config {config_path}"
        gateway_cmd += f" --config {config_path}"

    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    if wizard:
        console.print(f"  1. Chat: [cyan]{agent_cmd}[/cyan]")
        console.print(f"  2. Start gateway: [cyan]{gateway_cmd}[/cyan]")
    else:
        console.print("  1. Authenticate Codex: [cyan]nanobot codex login[/cyan]")
        console.print(f"  2. Chat: [cyan]{agent_cmd}[/cyan]")
    console.print(
        "\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]"
    )


def _merge_missing_defaults(existing: Any, defaults: Any) -> Any:
    """Recursively fill in missing values from defaults without overwriting user config."""
    if not isinstance(existing, dict) or not isinstance(defaults, dict):
        return existing

    merged = dict(existing)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
        else:
            merged[key] = _merge_missing_defaults(merged[key], value)
    return merged


def _onboard_plugins(config_path: Path) -> None:
    """Inject default config for discovered channels into config.json."""
    import json

    from nanobot.channels.registry import discover_all

    all_channels = discover_all()
    if not all_channels or not config_path.exists():
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    channels = data.setdefault("channels", {})
    for name, cls in all_channels.items():
        defaults = cls.default_config()
        if name not in channels:
            channels[name] = defaults
        else:
            channels[name] = _merge_missing_defaults(channels[name], defaults)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _configure_codex_profile_on_onboard(
    config: Config,
    workspace: Path,
    *,
    prompt_if_unset: bool,
) -> bool:
    """Apply or prompt for workspace-local Codex profile management.

    Returns True when the config object was updated in-memory.
    """
    from nanobot.providers.codex_profile import CodexProfileManager

    apply_profile = config.codex.use_workspace_profile
    changed = False

    if apply_profile is None and prompt_if_unset:
        apply_profile = typer.confirm(
            "Apply nanobot-managed Codex profile for Codex App Server in this workspace?",
            default=True,
        )
        config.codex.use_workspace_profile = apply_profile
        changed = True

    if apply_profile is None:
        return changed

    manager = CodexProfileManager(workspace, profile_name=config.codex.profile_name)
    if apply_profile:
        manager.ensure_profile()
        console.print(
            f"[green]✓[/green] Enabled Codex workspace profile "
            f"[cyan]{config.codex.profile_name}[/cyan]"
        )
        return changed

    removed = manager.remove_managed_profile()
    if removed:
        console.print("[green]✓[/green] Removed nanobot-managed Codex workspace profile files")
    else:
        console.print("[dim]Skipped Codex workspace profile setup[/dim]")
    return changed


def _make_provider(config: Config):
    """Create the Codex App Server runtime provider."""
    model = config.agents.defaults.model
    if not model:
        model = "openai-codex/gpt-5.3-codex"

    _apply_langsmith_config(config)

    try:
        from nanobot.providers.openai_codex_app_server_provider import (
            OpenAICodexAppServerProvider,
        )
    except ModuleNotFoundError as exc:
        console.print(
            "[red]Error: Codex App Server provider is unavailable in this build.[/red]"
        )
        raise typer.Exit(1) from exc

    return OpenAICodexAppServerProvider(
        default_model=model,
        workspace=config.workspace_path,
        profile_name=config.codex.profile_name,
        use_workspace_profile=config.codex.use_workspace_profile is not False,
        sandbox=config.codex.sandbox,
    )


def _apply_langsmith_config(config: Config) -> None:
    """Apply LangSmith runtime settings from config.json."""
    cfg = config.observability.langsmith
    if not cfg.enabled:
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if cfg.api_key:
        os.environ["LANGSMITH_API_KEY"] = cfg.api_key
    if cfg.project:
        os.environ["LANGSMITH_PROJECT"] = cfg.project
    if cfg.endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = cfg.endpoint


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int | None = typer.Option(None, "--port", "-p", help="Gateway port"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the nanobot gateway."""
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop
    from nanobot.channels.manager import ChannelManager
    from nanobot.session.manager import SessionManager
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.heartbeat.service import HeartbeatService
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    config, _ = _load_runtime_config(workspace=workspace, config=config)
    resolved_port = port if port is not None else config.gateway.port
    console.print(f"{__logo__} Starting nanobot gateway on port {resolved_port}...")
    sync_workspace_templates(config.workspace_path)
    config.channels.web.port = resolved_port
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)

    # Create cron service first (callback set after agent creation)
    _migrate_workspace_cron_store(config)
    cron_store_path = _cron_store_path()
    cron = CronService(cron_store_path)

    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        routing_enabled=config.agents.defaults.intent_execution_routing_enabled,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
    )

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        delivery_channel = job.payload.channel or "cli"
        delivery_chat_id, delivery_metadata = resolve_delivery_target(
            delivery_channel,
            job.payload.to,
        )
        # Run scheduled turns in an isolated AgentLoop so cron work cannot stall
        # behind the gateway's shared inbound-processing lock.
        cron_provider = _make_provider(config)
        cron_agent = AgentLoop(
            bus=bus,
            provider=cron_provider,
            workspace=config.workspace_path,
            model=config.agents.defaults.model,
            temperature=config.agents.defaults.temperature,
            max_tokens=config.agents.defaults.max_tokens,
            max_iterations=config.agents.defaults.max_tool_iterations,
            context_window_tokens=config.agents.defaults.context_window_tokens,
            reasoning_effort=config.agents.defaults.reasoning_effort,
            routing_enabled=config.agents.defaults.intent_execution_routing_enabled,
            brave_api_key=config.tools.web.search.api_key or None,
            exec_config=config.tools.exec,
            cron_service=cron,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            session_manager=session_manager,
            mcp_servers=config.tools.mcp_servers,
        )
        try:
            outbound = await cron_agent.process_direct(
                job.payload.message,
                session_key=f"cron:{job.id}",
                channel=delivery_channel,
                chat_id=delivery_chat_id,
                metadata=delivery_metadata,
            )
            response = outbound.content if outbound else ""
        finally:
            await cron_agent.close_mcp()
        if job.payload.deliver and job.payload.to:
            await bus.publish_outbound(
                OutboundMessage(
                    channel=delivery_channel,
                    chat_id=delivery_chat_id,
                    content=response or "",
                    metadata=delivery_metadata,
                )
            )
        return response

    cron.on_job = on_cron_job

    # Create channel manager
    channels = ChannelManager(config, bus)

    def _pick_heartbeat_target() -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages."""
        enabled = set(channels.enabled_channels)
        for item in session_manager.list_sessions():
            key = item.get("key") or ""
            if ":" not in key:
                continue
            channel, chat_id = key.split(":", 1)
            if channel in {"cli", "system"}:
                continue
            if channel in enabled and chat_id:
                return channel, chat_id
        return "cli", "direct"

    # Create heartbeat service
    async def on_heartbeat_execute(tasks: str) -> str:
        """Phase 2: execute heartbeat tasks through the full agent loop."""
        channel, chat_id = _pick_heartbeat_target()

        async def _silent(*_args, **_kwargs) -> None:
            return

        outbound = await agent.process_direct(
            tasks,
            session_key="heartbeat",
            channel=channel,
            chat_id=chat_id,
            on_progress=_silent,
        )
        return outbound.content if outbound else ""

    async def on_heartbeat_notify(response: str) -> None:
        """Deliver a heartbeat response to the user's channel."""
        from nanobot.bus.events import OutboundMessage

        channel, chat_id = _pick_heartbeat_target()
        if channel == "cli":
            return
        await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    hb_cfg = config.gateway.heartbeat
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        provider=provider,
        model=agent.model,
        on_execute=on_heartbeat_execute,
        on_notify=on_heartbeat_notify,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
    )

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    if config.channels.web.enabled:
        console.print(
            f"[green]✓[/green] Web UI: http://{config.channels.web.host}:{config.channels.web.port}/"
        )

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")

    async def run():
        agent_task: asyncio.Task[Any] | None = None
        channels_task: asyncio.Task[Any] | None = None
        try:
            await cron.start()
            await heartbeat.start()
            agent_task = asyncio.create_task(agent.run())
            channels_task = asyncio.create_task(channels.start_all())

            done, _pending = await asyncio.wait(
                {agent_task, channels_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    raise exc
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            heartbeat.stop()
            cron.stop()
            agent.stop()
            if agent.restart_requested:
                await channels.drain_outbound(timeout=1.0)
            await channels.stop_all()
            for task in (agent_task, channels_task):
                if task is not None and not task.done():
                    task.cancel()
            pending_tasks = [task for task in (agent_task, channels_task) if task is not None]
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await agent.close_mcp()

    asyncio.run(run())


# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    markdown: bool = typer.Option(
        True, "--markdown/--no-markdown", help="Render assistant output as Markdown"
    ),
    logs: bool = typer.Option(
        False, "--logs/--no-logs", help="Show nanobot runtime logs during chat"
    ),
):
    """Interact with the agent directly."""
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop
    from nanobot.cron.service import CronService
    from loguru import logger

    config, _ = _load_runtime_config(workspace=workspace, config=config)
    sync_workspace_templates(config.workspace_path)

    bus = MessageBus()
    provider = _make_provider(config)

    # Create cron service for tool usage (no callback needed for CLI unless running)
    _migrate_workspace_cron_store(config)
    cron_store_path = _cron_store_path()
    cron = CronService(cron_store_path)

    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        routing_enabled=config.agents.defaults.intent_execution_routing_enabled,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
    )

    # Show spinner when logs are off (no output to miss); skip when logs are on
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext

            return nullcontext()
        # Animated spinner is safe to use with prompt_toolkit input handling
        return console.status("[dim]nanobot is thinking...[/dim]", spinner="dots")

    async def _cli_progress(content: str = "", *, tool_hint: str | None = None) -> None:
        progress_text = (content or "").strip()
        hint_text = (tool_hint or "").strip()
        if not progress_text and not hint_text:
            return
        if progress_text:
            console.print(f"  [dim]↳ {progress_text}[/dim]")
            return
        console.print(f"  [dim]↳ {hint_text}[/dim]")

    if message:
        # Single message mode
        async def run_once():
            with _thinking_ctx():
                outbound = await agent_loop.process_direct(
                    message,
                    session_id,
                    on_progress=_cli_progress,
                )
            _print_agent_response(
                outbound.content if outbound else "",
                render_markdown=markdown,
                metadata=outbound.metadata if outbound else None,
            )
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode
        _init_prompt_session()
        console.print(
            f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n"
        )

        def _exit_on_sigint(signum, frame):
            _restore_terminal()
            console.print("\nGoodbye!")
            os._exit(0)

        signal.signal(signal.SIGINT, _exit_on_sigint)

        async def run_interactive():
            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        with _thinking_ctx():
                            outbound = await agent_loop.process_direct(
                                user_input,
                                session_id,
                                on_progress=_cli_progress,
                            )
                        _print_agent_response(
                            outbound.content if outbound else "",
                            render_markdown=markdown,
                            metadata=outbound.metadata if outbound else None,
                        )
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from nanobot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("Configuration", style="yellow")

    # WhatsApp
    wa = config.channels.whatsapp
    table.add_row("WhatsApp", "✓" if wa.enabled else "✗", wa.bridge_url)

    dc = config.channels.discord
    table.add_row("Discord", "✓" if dc.enabled else "✗", dc.gateway_url)

    # Feishu
    fs = config.channels.feishu
    fs_config = f"app_id: {fs.app_id[:10]}..." if fs.app_id else "[dim]not configured[/dim]"
    table.add_row("Feishu", "✓" if fs.enabled else "✗", fs_config)

    # Mochat
    mc = config.channels.mochat
    mc_base = mc.base_url or "[dim]not configured[/dim]"
    table.add_row("Mochat", "✓" if mc.enabled else "✗", mc_base)

    # Telegram
    tg = config.channels.telegram
    tg_config = f"token: {tg.token[:10]}..." if tg.token else "[dim]not configured[/dim]"
    table.add_row("Telegram", "✓" if tg.enabled else "✗", tg_config)

    # Slack
    slack = config.channels.slack
    slack_config = "socket" if slack.app_token and slack.bot_token else "[dim]not configured[/dim]"
    table.add_row("Slack", "✓" if slack.enabled else "✗", slack_config)

    web = config.channels.web
    table.add_row("Web", "✓" if web.enabled else "✗", f"http://{web.host}:{web.port}")

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess

    # User's bridge location
    user_bridge = Path.home() / ".nanobot" / "bridge"

    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge

    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)

    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # nanobot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)

    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge

    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall nanobot")
        raise typer.Exit(1)

    console.print(f"{__logo__} Setting up bridge...")

    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))

    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=user_bridge, check=True, capture_output=True)

        console.print("  Building...")
        subprocess.run(["npm", "run", "build"], cwd=user_bridge, check=True, capture_output=True)

        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)

    return user_bridge


@channels_app.command("login")
def channels_login(
    channel: str = typer.Argument(..., help="Channel name to log in"),
):
    """Link device via QR code."""
    import subprocess
    from nanobot.config.loader import load_config

    if channel.lower() != "whatsapp":
        console.print(f"[red]Channel login is not supported for {channel}.[/red]")
        raise typer.Exit(1)

    config = load_config()
    bridge_dir = _get_bridge_dir()

    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")

    env = {**os.environ}
    if config.channels.whatsapp.bridge_token:
        env["BRIDGE_TOKEN"] = config.channels.whatsapp.bridge_token

    try:
        subprocess.run(["npm", "start"], cwd=bridge_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]npm not found. Please install Node.js.[/red]")


# ============================================================================
# Cron Commands
# ============================================================================

cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all: bool = typer.Option(False, "--all", "-a", help="Include disabled jobs"),
):
    """List scheduled jobs."""
    from nanobot.cron.service import CronService

    store_path = _cron_store_path()
    service = CronService(store_path)

    jobs = service.list_jobs(include_disabled=all)

    if not jobs:
        console.print("No scheduled jobs.")
        return

    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("Status")
    table.add_column("Next Run")

    import time
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo

    for job in jobs:
        # Format schedule
        if job.schedule.kind == "every":
            sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
        elif job.schedule.kind == "cron":
            sched = (
                f"{job.schedule.expr or ''} ({job.schedule.tz})"
                if job.schedule.tz
                else (job.schedule.expr or "")
            )
        else:
            sched = "one-time"

        # Format next run
        next_run = ""
        if job.state.next_run_at_ms:
            ts = job.state.next_run_at_ms / 1000
            try:
                tz = ZoneInfo(job.schedule.tz) if job.schedule.tz else None
                next_run = _dt.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M")
            except Exception:
                next_run = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))

        status = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"

        table.add_row(job.id, job.name, sched, status, next_run)

    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    message: str = typer.Option(..., "--message", "-m", help="Message for agent"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    cron_expr: str = typer.Option(None, "--cron", "-c", help="Cron expression (e.g. '0 9 * * *')"),
    tz: str | None = typer.Option(
        None, "--tz", help="IANA timezone for cron (e.g. 'America/Vancouver')"
    ),
    at: str = typer.Option(None, "--at", help="Run once at time (ISO format)"),
    deliver: bool = typer.Option(False, "--deliver", "-d", help="Deliver response to channel"),
    to: str = typer.Option(None, "--to", help="Recipient for delivery"),
    channel: str = typer.Option(
        None, "--channel", help="Channel for delivery (e.g. 'telegram', 'whatsapp')"
    ),
):
    """Add a scheduled job."""
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronSchedule

    if tz and not cron_expr:
        console.print("[red]Error: --tz can only be used with --cron[/red]")
        raise typer.Exit(1)

    # Determine schedule type
    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
    elif at:
        import datetime

        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Error: Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)

    store_path = _cron_store_path()
    service = CronService(store_path)

    try:
        job = service.add_job(
            name=name,
            schedule=schedule,
            message=message,
            deliver=deliver,
            to=to,
            channel=channel,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]✓[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
):
    """Remove a scheduled job."""
    from nanobot.cron.service import CronService

    store_path = _cron_store_path()
    service = CronService(store_path)

    if service.remove_job(job_id):
        console.print(f"[green]✓[/green] Removed job {job_id}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID"),
    disable: bool = typer.Option(False, "--disable", help="Disable instead of enable"),
):
    """Enable or disable a job."""
    from nanobot.cron.service import CronService

    store_path = _cron_store_path()
    service = CronService(store_path)

    job = service.enable_job(job_id, enabled=not disable)
    if job:
        status = "disabled" if disable else "enabled"
        console.print(f"[green]✓[/green] Job '{job.name}' {status}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    force: bool = typer.Option(False, "--force", "-f", help="Run even if disabled"),
):
    """Manually run a job."""
    from loguru import logger
    from nanobot.config.loader import load_config
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop

    logger.disable("nanobot")

    config = load_config()
    sync_workspace_templates(config.workspace_path)
    provider = _make_provider(config)
    bus = MessageBus()

    store_path = _cron_store_path()
    service = CronService(store_path)
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        routing_enabled=config.agents.defaults.intent_execution_routing_enabled,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=service,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
    )

    result_holder: list[str] = []

    async def on_job(job: CronJob) -> str | None:
        delivery_channel = job.payload.channel or "cli"
        delivery_chat_id, delivery_metadata = resolve_delivery_target(
            delivery_channel,
            job.payload.to,
        )
        outbound = await agent_loop.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=delivery_channel,
            chat_id=delivery_chat_id,
            metadata=delivery_metadata,
        )
        response = outbound.content if outbound else ""
        result_holder.append(response)
        return response

    service.on_job = on_job

    async def run():
        try:
            return await service.run_job(job_id, force=force)
        finally:
            await agent_loop.close_mcp()

    if asyncio.run(run()):
        console.print(f"[green]✓[/green] Job executed")
        if result_holder:
            _print_agent_response(result_holder[0], render_markdown=True)
    else:
        console.print(f"[red]Failed to run job {job_id}[/red]")


# ============================================================================
# Status Commands
# ============================================================================

@app.command()
def status():
    """Show nanobot status."""
    from nanobot.config.loader import load_config, get_config_path

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(
        f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}"
    )
    console.print(
        f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}"
    )

    if config_path.exists():
        console.print(f"Model: {config.agents.defaults.model}")
        console.print("Runtime: Codex App Server")
        profile_state = config.codex.use_workspace_profile
        if profile_state is None:
            console.print("Codex workspace profile: [dim]auto[/dim]")
        else:
            console.print(
                "Codex workspace profile: "
                f"{'[green]enabled[/green]' if profile_state else '[dim]disabled[/dim]'}"
            )
        console.print(f"Codex profile name: {config.codex.profile_name}")
        console.print(f"Codex sandbox: {config.codex.sandbox}")
        console.print(f"Codex auth: {_get_codex_auth_label()}")


# ============================================================================
# Codex Runtime
# ============================================================================

codex_app = typer.Typer(help="Manage Codex runtime")
app.add_typer(codex_app, name="codex")

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")


_LOGIN_HANDLERS: dict[str, callable] = {}


def _register_login(name: str):
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn

    return decorator


def _get_codex_auth_label() -> str:
    """Return a short label describing Codex OAuth availability."""
    try:
        from oauth_cli_kit import get_token

        token = get_token()
    except Exception:
        token = None

    if token and getattr(token, "access", None):
        account_id = getattr(token, "account_id", "")
        if account_id:
            return f"[green]✓[/green] {account_id}"
        return "[green]✓[/green] authenticated"
    return "[dim]not authenticated[/dim]"


@codex_app.command("login")
def codex_login():
    """Authenticate with Codex OAuth."""
    console.print(f"{__logo__} Codex Login\n")
    _login_openai_codex()


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(
        ..., help="OAuth provider (supported: 'openai-codex')"
    ),
):
    """Authenticate with an OAuth provider."""
    key = provider.replace("-", "_")
    if key != "openai_codex":
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: openai-codex")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get("openai_codex")
    if not handler:
        console.print("[red]Login not implemented for OpenAI Codex[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - OpenAI Codex\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive

        token = None
        try:
            token = get_token()
        except Exception:
            pass
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = login_oauth_interactive(
                print_fn=lambda s: console.print(s),
                prompt_fn=lambda s: typer.prompt(s),
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]"
        )
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
