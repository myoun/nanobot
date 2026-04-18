from __future__ import annotations

from pathlib import Path

from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.memory import MemoryTool


async def _exec(tool: MemoryTool, **kwargs: object) -> str:
    return await tool.execute(**kwargs)


def test_memory_tool_creates_and_reads_global_preference(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    store = MemoryStore(workspace)
    tool = MemoryTool(store)

    import asyncio

    create_result = asyncio.run(
        _exec(
            tool,
            action="create",
            scope="global",
            kind="preferences",
            title="Logs First",
            content="에러 로그를 먼저 본다.",
        )
    )
    assert "Memory saved." in create_result
    assert "global/preferences/" in create_result

    list_result = asyncio.run(_exec(tool, action="list", scope="global", kind="preferences"))
    assert "Logs First" in list_result

    item_id = next(store.global_preferences_dir.glob("*.md")).relative_to(store.item_root).as_posix()
    read_result = asyncio.run(_exec(tool, action="read", item_id=item_id))
    assert "에러 로그를 먼저 본다." in read_result


def test_memory_tool_updates_existing_item(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    store = MemoryStore(workspace)
    tool = MemoryTool(store)

    import asyncio

    asyncio.run(
        _exec(
            tool,
            action="create",
            scope="workspace",
            kind="rules",
            title="Tests First",
            content="테스트를 먼저 돌린다.",
        )
    )

    item_path = next(store.workspace_rules_dir.glob("*.md"))
    item_id = item_path.relative_to(store.item_root).as_posix()
    update_result = asyncio.run(
        _exec(
            tool,
            action="update",
            item_id=item_id,
            content="이 워크스페이스에서는 테스트와 린트를 먼저 돌린다.",
        )
    )

    assert "Memory updated." in update_result
    assert "린트" in item_path.read_text(encoding="utf-8")
