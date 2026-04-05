import asyncio
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from nanobot.bus.events import OutboundMessage
from nanobot.cli.commands import _make_provider, app
from nanobot.config.schema import Config
from nanobot.providers.openai_codex_app_server_provider import OpenAICodexAppServerProvider
from nanobot.providers.openai_codex_provider import _strip_model_prefix

runner = CliRunner()


class _StopGatewayError(RuntimeError):
    pass


import shutil

import pytest


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.cli.commands.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_lc.side_effect = lambda _config_path=None: Config()

        def _save_config(config: Config, config_path: Path | None = None):
            target = config_path or config_file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(config.model_dump(by_alias=True)), encoding="utf-8")

        mock_sc.side_effect = _save_config

        yield config_file, workspace_dir, mock_ws

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir, mock_ws = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert "nanobot codex login" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()
    expected_workspace = Config().workspace_path
    assert mock_ws.call_args.args == (expected_workspace,)


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir, _ = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir, _ = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir, _ = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def _strip_ansi(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def test_onboard_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["onboard", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output
    assert "--wizard" in stripped_output
    assert "--dir" not in stripped_output


def test_onboard_interactive_discard_does_not_save_or_create_workspace(mock_paths, monkeypatch):
    config_file, workspace_dir, _ = mock_paths

    from nanobot.cli.onboard import OnboardResult

    monkeypatch.setattr(
        "nanobot.cli.onboard.run_onboard",
        lambda initial_config: OnboardResult(config=initial_config, should_save=False),
    )

    result = runner.invoke(app, ["onboard", "--wizard"])

    assert result.exit_code == 0
    assert "No changes were saved" in result.stdout
    assert not config_file.exists()
    assert not workspace_dir.exists()


def test_onboard_uses_explicit_config_and_workspace_paths(tmp_path, monkeypatch):
    config_path = tmp_path / "instance" / "config.json"
    workspace_path = tmp_path / "workspace"

    monkeypatch.setattr("nanobot.channels.registry.discover_all", lambda: {})

    result = runner.invoke(
        app,
        ["onboard", "--config", str(config_path), "--workspace", str(workspace_path)],
    )

    assert result.exit_code == 0
    saved = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    assert saved.workspace_path == workspace_path
    assert (workspace_path / "AGENTS.md").exists()
    stripped_output = _strip_ansi(result.stdout)
    compact_output = stripped_output.replace("\n", "")
    resolved_config = str(config_path.resolve())
    assert resolved_config in compact_output
    assert f"--config {resolved_config}" in compact_output


def test_onboard_wizard_preserves_explicit_config_in_next_steps(tmp_path, monkeypatch):
    config_path = tmp_path / "instance" / "config.json"
    workspace_path = tmp_path / "workspace"

    from nanobot.cli.onboard import OnboardResult

    monkeypatch.setattr(
        "nanobot.cli.onboard.run_onboard",
        lambda initial_config: OnboardResult(config=initial_config, should_save=True),
    )
    monkeypatch.setattr("nanobot.channels.registry.discover_all", lambda: {})

    result = runner.invoke(
        app,
        ["onboard", "--wizard", "--config", str(config_path), "--workspace", str(workspace_path)],
    )

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    compact_output = stripped_output.replace("\n", "")
    resolved_config = str(config_path.resolve())
    assert f'nanobot agent -m "Hello!" --config {resolved_config}' in compact_output
    assert f"nanobot gateway --config {resolved_config}" in compact_output


def test_config_defaults_to_codex_model():
    config = Config()

    assert config.agents.defaults.model == "openai-codex/gpt-5.3-codex"


def test_config_dump_omits_legacy_providers_block():
    config = Config()

    dumped = config.model_dump(by_alias=True)

    assert "providers" not in dumped


def test_openai_codex_strip_prefix_supports_hyphen_and_underscore():
    assert _strip_model_prefix("openai-codex/gpt-5.1-codex") == "gpt-5.1-codex"
    assert _strip_model_prefix("openai_codex/gpt-5.1-codex") == "gpt-5.1-codex"


def test_make_provider_returns_codex_app_server_provider(tmp_path: Path):
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    config.codex.profile_name = "workspace-profile"
    config.codex.use_workspace_profile = False
    config.codex.sandbox = "read-only"

    provider = _make_provider(config)

    assert isinstance(provider, OpenAICodexAppServerProvider)
    assert provider.get_default_model() == config.agents.defaults.model
    assert provider.workspace == str(config.workspace_path.resolve())
    assert provider.sandbox == "read-only"


@pytest.fixture
def mock_agent_runtime(tmp_path):
    """Mock agent command dependencies for focused CLI tests."""
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "default-workspace")

    with patch("nanobot.config.loader.load_config", return_value=config) as mock_load_config, \
         patch("nanobot.cli.commands.sync_workspace_templates") as mock_sync_templates, \
         patch("nanobot.cli.commands._make_provider", return_value=object()), \
         patch("nanobot.cli.commands._print_agent_response") as mock_print_response, \
         patch("nanobot.bus.queue.MessageBus"), \
         patch("nanobot.cron.service.CronService"), \
         patch("nanobot.agent.loop.AgentLoop") as mock_agent_loop_cls:

        agent_loop = MagicMock()
        agent_loop.channels_config = None
        agent_loop.process_direct = AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="direct", content="mock-response"),
        )
        agent_loop.close_mcp = AsyncMock(return_value=None)
        mock_agent_loop_cls.return_value = agent_loop

        yield {
            "config": config,
            "load_config": mock_load_config,
            "sync_templates": mock_sync_templates,
            "agent_loop_cls": mock_agent_loop_cls,
            "agent_loop": agent_loop,
            "print_response": mock_print_response,
        }


def test_agent_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["agent", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output


def test_agent_uses_default_config_when_no_workspace_or_config_flags(mock_agent_runtime):
    result = runner.invoke(app, ["agent", "-m", "hello"])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (None,)
    assert mock_agent_runtime["sync_templates"].call_args.args == (
        mock_agent_runtime["config"].workspace_path,
    )
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == (
        mock_agent_runtime["config"].workspace_path
    )
    mock_agent_runtime["agent_loop"].process_direct.assert_awaited_once()
    mock_agent_runtime["print_response"].assert_called_once_with(
        "mock-response", render_markdown=True, metadata={},
    )


def test_agent_uses_explicit_config_path(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)


def test_agent_config_sets_active_path(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "nanobot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.cron.service.CronService", lambda _store: object())

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs):
            return OutboundMessage(channel="cli", chat_id="direct", content="ok")

        async def close_mcp(self) -> None:
            return None

    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.cli.commands._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["config_path"] == config_file.resolve()


def test_agent_uses_instance_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "agent-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: tmp_path / "instance" / "cron")

    class _FakeCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs):
            return OutboundMessage(channel="cli", chat_id="direct", content="ok")

        async def close_mcp(self) -> None:
            return None

    monkeypatch.setattr("nanobot.cron.service.CronService", _FakeCron)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.cli.commands._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["cron_store"] == tmp_path / "instance" / "cron" / "jobs.json"


def test_agent_workspace_override_still_uses_instance_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    override = tmp_path / "override-workspace"
    config = Config()
    seen: dict[str, Path] = {}
    instance_cron_dir = tmp_path / "instance" / "cron"

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: instance_cron_dir)

    class _FakeCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs):
            return OutboundMessage(channel="cli", chat_id="direct", content="ok")

        async def close_mcp(self) -> None:
            return None

    monkeypatch.setattr("nanobot.cron.service.CronService", _FakeCron)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.cli.commands._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(
        app,
        ["agent", "-m", "hello", "-c", str(config_file), "-w", str(override)],
    )

    assert result.exit_code == 0
    assert seen["cron_store"] == instance_cron_dir / "jobs.json"
    assert not (override / "cron" / "jobs.json").exists()


def test_agent_custom_workspace_still_uses_instance_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    custom_workspace = tmp_path / "custom-workspace"
    config = Config()
    config.agents.defaults.workspace = str(custom_workspace)
    seen: dict[str, Path] = {}
    instance_cron_dir = tmp_path / "instance" / "cron"

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: instance_cron_dir)

    class _FakeCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs):
            return OutboundMessage(channel="cli", chat_id="direct", content="ok")

        async def close_mcp(self) -> None:
            return None

    monkeypatch.setattr("nanobot.cron.service.CronService", _FakeCron)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.cli.commands._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["cron_store"] == instance_cron_dir / "jobs.json"
    assert not (custom_workspace / "cron" / "jobs.json").exists()


def test_agent_overrides_workspace_path(mock_agent_runtime):
    workspace_path = Path("/tmp/agent-workspace")

    result = runner.invoke(app, ["agent", "-m", "hello", "-w", str(workspace_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == workspace_path


def test_agent_workspace_override_wins_over_config_workspace(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")
    workspace_path = Path("/tmp/agent-workspace")

    result = runner.invoke(
        app,
        ["agent", "-m", "hello", "-c", str(config_path), "-w", str(workspace_path)],
    )

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == workspace_path


def test_agent_hints_about_deprecated_memory_window(mock_agent_runtime, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"agents": {"defaults": {"memoryWindow": 42}}}))

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert "memoryWindow" in result.stdout
    assert "no longer used" in result.stdout


def test_heartbeat_retains_recent_messages_by_default():
    config = Config()

    assert config.gateway.heartbeat.keep_recent_messages == 8


def test_gateway_uses_workspace_from_config_by_default(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "nanobot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "nanobot.cli.commands.sync_workspace_templates",
        lambda path: seen.__setitem__("workspace", path),
    )
    monkeypatch.setattr(
        "nanobot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["config_path"] == config_file.resolve()
    assert seen["workspace"] == Path(config.agents.defaults.workspace)


def test_gateway_workspace_option_overrides_config(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    override = tmp_path / "override-workspace"
    seen: dict[str, Path] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "nanobot.cli.commands.sync_workspace_templates",
        lambda path: seen.__setitem__("workspace", path),
    )
    monkeypatch.setattr(
        "nanobot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(
        app,
        ["gateway", "--config", str(config_file), "--workspace", str(override)],
    )

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["workspace"] == override
    assert config.workspace_path == override


def test_gateway_uses_instance_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.session.manager.SessionManager", lambda _workspace: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: tmp_path / "instance" / "cron")

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    monkeypatch.setattr("nanobot.cron.service.CronService", _StopCron)

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == tmp_path / "instance" / "cron" / "jobs.json"


def test_gateway_workspace_override_still_uses_instance_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    override = tmp_path / "override-workspace"
    config = Config()
    seen: dict[str, Path] = {}
    instance_cron_dir = tmp_path / "instance" / "cron"

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.session.manager.SessionManager", lambda _workspace: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: instance_cron_dir)

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    monkeypatch.setattr("nanobot.cron.service.CronService", _StopCron)

    result = runner.invoke(
        app,
        ["gateway", "--config", str(config_file), "--workspace", str(override)],
    )

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == instance_cron_dir / "jobs.json"
    assert not (override / "cron" / "jobs.json").exists()


def test_gateway_custom_workspace_still_uses_instance_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    custom_workspace = tmp_path / "custom-workspace"
    config = Config()
    config.agents.defaults.workspace = str(custom_workspace)
    seen: dict[str, Path] = {}
    instance_cron_dir = tmp_path / "instance" / "cron"

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.session.manager.SessionManager", lambda _workspace: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: instance_cron_dir)

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    monkeypatch.setattr("nanobot.cron.service.CronService", _StopCron)

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == instance_cron_dir / "jobs.json"
    assert not (custom_workspace / "cron" / "jobs.json").exists()


def test_migrate_workspace_cron_store_moves_workspace_file_into_instance_store(tmp_path: Path) -> None:
    """Legacy workspace jobs.json is moved into the instance cron store on first run."""
    from nanobot.cli.commands import _migrate_workspace_cron_store

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    workspace_cron = config.workspace_path / "cron" / "jobs.json"
    workspace_cron.parent.mkdir(parents=True)
    workspace_cron.write_text('{"jobs": []}')

    instance_dir = tmp_path / "instance" / "cron"
    instance_cron = instance_dir / "jobs.json"

    with patch("nanobot.config.paths.get_cron_dir", return_value=instance_dir):
        _migrate_workspace_cron_store(config)

    assert instance_cron.exists()
    assert instance_cron.read_text() == '{"jobs": []}'
    assert not workspace_cron.exists()


def test_migrate_workspace_cron_store_merges_workspace_jobs_into_instance_store(tmp_path: Path) -> None:
    """Migration preserves existing instance jobs while importing workspace-only jobs."""
    from nanobot.cli.commands import _migrate_workspace_cron_store

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    workspace_cron = config.workspace_path / "cron" / "jobs.json"
    workspace_cron.parent.mkdir(parents=True)
    workspace_cron.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    {"id": "workspace-only", "updatedAtMs": 2},
                    {"id": "shared", "updatedAtMs": 5, "payload": {"message": "workspace"}},
                ],
            }
        )
    )

    instance_dir = tmp_path / "instance" / "cron"
    instance_dir.mkdir(parents=True)
    instance_cron = instance_dir / "jobs.json"
    instance_cron.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    {"id": "instance-only", "updatedAtMs": 1},
                    {"id": "shared", "updatedAtMs": 3, "payload": {"message": "instance"}},
                ],
            }
        )
    )

    with patch("nanobot.config.paths.get_cron_dir", return_value=instance_dir):
        _migrate_workspace_cron_store(config)

    merged = json.loads(instance_cron.read_text())
    assert [job["id"] for job in merged["jobs"]] == ["instance-only", "shared", "workspace-only"]
    assert merged["jobs"][1]["payload"]["message"] == "workspace"
    assert not workspace_cron.exists()


def test_gateway_uses_configured_port_when_cli_flag_is_missing(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.gateway.port = 18791

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr(
        "nanobot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert "port 18791" in result.stdout


def test_gateway_cli_port_overrides_configured_port(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.gateway.port = 18791

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr(
        "nanobot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file), "--port", "18792"])

    assert isinstance(result.exception, _StopGatewayError)
    assert "port 18792" in result.stdout


def test_gateway_gracefully_stops_and_exits_after_restart_request(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.channels.web.enabled = False
    seen: dict[str, object] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: tmp_path / "instance" / "cron")

    class _FakeBus:
        pass

    class _FakeSessionManager:
        def __init__(self, _workspace) -> None:
            pass

        def list_sessions(self):
            return []

    class _FakeCron:
        def __init__(self, _store_path: Path) -> None:
            self.on_job = None

        async def start(self) -> None:
            seen["cron_started"] = True

        def stop(self) -> None:
            seen["cron_stopped"] = True

        def status(self) -> dict[str, int]:
            return {"jobs": 0}

    class _FakeHeartbeat:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def start(self) -> None:
            seen["heartbeat_started"] = True

        def stop(self) -> None:
            seen["heartbeat_stopped"] = True

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            self.model = "test-model"
            self.restart_requested = False

        async def run(self) -> None:
            self.restart_requested = True

        async def close_mcp(self) -> None:
            seen["agent_closed"] = True

        def stop(self) -> None:
            seen["agent_stopped"] = True

        async def process_direct(self, *args, **kwargs):
            return None

    class _FakeChannels:
        def __init__(self, _config, _bus) -> None:
            self.enabled_channels = []
            self._stop_event = asyncio.Event()

        async def start_all(self) -> None:
            seen["channels_started"] = True
            await self._stop_event.wait()

        async def stop_all(self) -> None:
            seen["channels_stopped"] = True
            self._stop_event.set()

        async def drain_outbound(self, timeout: float = 1.0) -> None:
            seen["drain_timeout"] = timeout

    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _FakeBus)
    monkeypatch.setattr("nanobot.session.manager.SessionManager", _FakeSessionManager)
    monkeypatch.setattr("nanobot.cron.service.CronService", _FakeCron)
    monkeypatch.setattr("nanobot.heartbeat.service.HeartbeatService", _FakeHeartbeat)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", _FakeChannels)

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert result.exit_code == 0
    assert seen["cron_started"] is True
    assert seen["heartbeat_started"] is True
    assert seen["drain_timeout"] == 1.0
    assert seen["channels_stopped"] is True
    assert seen["agent_stopped"] is True
    assert seen["agent_closed"] is True


def test_gateway_cron_jobs_use_isolated_agent_loop(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.channels.web.enabled = False
    seen: dict[str, object] = {"agents": [], "process_calls": []}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: tmp_path / "instance" / "cron")

    class _FakeBus:
        async def publish_outbound(self, msg) -> None:
            seen["outbound"] = msg

    class _FakeSessionManager:
        def __init__(self, _workspace) -> None:
            pass

        def list_sessions(self):
            return []

    class _FakeCron:
        def __init__(self, _store_path: Path) -> None:
            self.on_job = None

        async def start(self) -> None:
            from types import SimpleNamespace

            assert self.on_job is not None
            job = SimpleNamespace(
                id="job-1",
                payload=SimpleNamespace(
                    message="hello",
                    channel="telegram",
                    to="-1003672808487:topic:3",
                    deliver=True,
                ),
            )
            await self.on_job(job)
            raise _StopGatewayError("stop")

        def stop(self) -> None:
            seen["cron_stopped"] = True

        def status(self) -> dict[str, int]:
            return {"jobs": 1}

    class _FakeHeartbeat:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def start(self) -> None:
            seen["heartbeat_started"] = True

        def stop(self) -> None:
            seen["heartbeat_stopped"] = True

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            self.model = "test-model"
            self.restart_requested = False
            cast_agents = seen["agents"]
            assert isinstance(cast_agents, list)
            cast_agents.append(self)

        async def run(self) -> None:
            return None

        async def close_mcp(self) -> None:
            seen["closed_count"] = int(seen.get("closed_count", 0)) + 1

        def stop(self) -> None:
            seen["agent_stopped"] = True

        async def process_direct(self, *args, **kwargs):
            cast_calls = seen["process_calls"]
            assert isinstance(cast_calls, list)
            cast_calls.append(self)
            return OutboundMessage(
                channel=kwargs["channel"],
                chat_id=kwargs["chat_id"],
                content="cron ok",
                metadata=kwargs.get("metadata") or {},
            )

    class _FakeChannels:
        def __init__(self, _config, _bus) -> None:
            self.enabled_channels = []

        async def start_all(self) -> None:
            return None

        async def stop_all(self) -> None:
            seen["channels_stopped"] = True

        async def drain_outbound(self, timeout: float = 1.0) -> None:
            seen["drain_timeout"] = timeout

    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _FakeBus)
    monkeypatch.setattr("nanobot.session.manager.SessionManager", _FakeSessionManager)
    monkeypatch.setattr("nanobot.cron.service.CronService", _FakeCron)
    monkeypatch.setattr("nanobot.heartbeat.service.HeartbeatService", _FakeHeartbeat)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", _FakeChannels)

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    agents = seen["agents"]
    calls = seen["process_calls"]
    assert isinstance(agents, list)
    assert isinstance(calls, list)
    assert len(agents) == 2
    assert calls == [agents[1]]
    outbound = seen["outbound"]
    assert isinstance(outbound, OutboundMessage)
    assert outbound.chat_id == "-1003672808487"
    assert outbound.metadata == {"message_thread_id": 3}


def test_channels_login_requires_channel_name() -> None:
    result = runner.invoke(app, ["channels", "login"])

    assert result.exit_code == 2
