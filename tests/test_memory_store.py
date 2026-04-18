from __future__ import annotations

from pathlib import Path

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore


def test_memory_store_reads_itemized_memory_scopes_and_legacy_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    store = MemoryStore(workspace)
    store.memory_file.write_text("Legacy project memory.", encoding="utf-8")
    (store.global_instructions_dir / "language.md").write_text(
        "항상 한국어로 답변한다.",
        encoding="utf-8",
    )
    (store.global_preferences_dir / "logs-first.md").write_text(
        "에러 로그를 먼저 본다.",
        encoding="utf-8",
    )
    (store.workspace_rules_dir / "tests-first.md").write_text(
        "이 워크스페이스에서는 테스트를 먼저 돌린다.",
        encoding="utf-8",
    )
    (store.workspace_memory_dir / "project.md").write_text(
        "현재 프로젝트는 Codex App Server 기반이다.",
        encoding="utf-8",
    )

    context = store.get_memory_context()

    assert "## Global Instructions" in context
    assert "항상 한국어로 답변한다." in context
    assert "## Workspace Rules" in context
    assert "테스트를 먼저 돌린다." in context
    assert "## Workspace Memory" in context
    assert "Codex App Server 기반" in context
    assert "## Global Memory" in context
    assert "에러 로그를 먼저 본다." in context
    assert "## Legacy Workspace Memory" in context
    assert "Legacy project memory." in context


def test_context_builder_includes_itemized_memory_sections(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    store = MemoryStore(workspace)
    (store.global_instructions_dir / "language.md").write_text(
        "항상 한국어로 답변한다.",
        encoding="utf-8",
    )
    (store.workspace_rules_dir / "tests-first.md").write_text(
        "이 워크스페이스에서는 테스트를 먼저 돌린다.",
        encoding="utf-8",
    )

    builder = ContextBuilder(workspace)
    prompt = builder.build_app_server_prompt()

    assert "# Memory" in prompt
    assert "## Global Instructions" in prompt
    assert "항상 한국어로 답변한다." in prompt
    assert "## Workspace Rules" in prompt
    assert "테스트를 먼저 돌린다." in prompt


def test_memory_store_search_items_uses_itemized_index(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    store = MemoryStore(workspace)
    item_path = store.global_facts_dir / "nickname.md"
    item_path.write_text(
        "# Preferred Name\n\n닉네임은 nano다.\n",
        encoding="utf-8",
    )

    hits = store.search_items("nano", limit=5)
    item = store.read_item("global/facts/nickname.md")

    assert hits
    assert hits[0]["item_id"] == "global/facts/nickname.md"
    assert hits[0]["scope"] == "global"
    assert hits[0]["kind"] == "facts"
    assert "nano" in hits[0]["snippet"].lower()
    assert item is not None
    assert "닉네임은 nano다." in item["content"]


def test_memory_store_does_not_migrate_workspace_memory_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    legacy_dir = workspace / "memory"
    legacy_dir.mkdir(parents=True)
    legacy_memory = legacy_dir / "MEMORY.md"
    legacy_memory.write_text("Old workspace memory.", encoding="utf-8")

    store = MemoryStore(workspace)

    assert legacy_memory.exists()
    assert store.read_long_term() == ""
