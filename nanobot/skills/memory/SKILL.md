---
name: memory
description: Itemized long-term memory with explicit create/search/read/update workflows.
always: true
---

# Memory

## Structure

- Durable memory lives under `~/.nanobot/memories/`.
- Global instructions: `~/.nanobot/memories/global/instructions/*.md`
- Global facts: `~/.nanobot/memories/global/facts/*.md`
- Global preferences: `~/.nanobot/memories/global/preferences/*.md`
- Workspace rules: `~/.nanobot/memories/workspaces/<workspace>/rules/*.md`
- Workspace memory: `~/.nanobot/memories/workspaces/<workspace>/memory/*.md`

## How to Use It

- Use the `memory` tool for all durable memory changes.
- Create: save a new durable preference, fact, instruction, or workspace rule.
- Search/list/read: inspect existing memory before creating duplicates.
- Update: revise an existing item by `item_id`.

## Rules

- Do not create `workspace/memory/`.
- Do not store durable memory under `.codex/`.
- Do not invent your own long-term memory file path when the `memory` tool can do it.
- Session continuity belongs in `working_set.md`, `summary.md`, and `transcript.md`, not in global memory.
