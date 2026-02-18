# Session Operations Guide

This guide explains how to operate multi-session conversations per channel after session switching support.

## Scope

- Session lifecycle: create, switch, rename, pin, search, delete, restore
- Channel-specific UX patterns for Web, Telegram, and Discord
- Safe deletion workflow with trash/restore
- Auto-title policy behavior

## Core Concepts

- `conversation_key`: one chat in one channel (e.g. `telegram:12345`, `discord:98765`, `web:abcdef`)
- `session_id`: one conversation thread inside the same chat
- Active session: incoming messages are routed to this session until switched
- Pinned session: appears first in listings for quicker access

## Common Commands

These commands are available through the agent loop in text-capable channels.

```text
/new
/sessions
/session list
/session current
/session new [title]
/session switch <id|index|active>
/session rename <id|index|active> <new title>
/session pin <id|index|active>
/session unpin <id|index|active>
/session search <keyword>
/session delete <id|index|active>
/session trash
/session restore <id>
```

## Channel UX

### Web UI

- Session sidebar supports:
  - Create (`New`)
  - Switch (click session item)
  - Rename/Delete (buttons)
  - Search (search input)
  - Pin/Unpin (right-click session item)
- Session switch loads session history into the message area.
- Large histories are window-rendered for performance (`Load older messages`).

### Telegram

- `/sessions` opens inline keyboard session switcher.
- Buttons support:
  - Switch to listed session
  - New session
  - Refresh
  - Page navigation
  - Pin/Unpin active session
- If needed, use command fallback for search/trash/restore:
  - `/session search <keyword>`
  - `/session trash`
  - `/session restore <id>`

### Discord

- Sending `/sessions` creates a component panel message.
- Panel supports:
  - Select menu to switch sessions
  - New, Refresh
  - Pin/Unpin active session
  - Pagination buttons
- Search/trash/restore currently use text commands.

## Safe Deletion (Trash First)

- Deleting a session no longer immediately hard-deletes metadata from user flow.
- Deleted sessions are recoverable via trash:
  - View: `/session trash`
  - Restore: `/session restore <id>`
- If the last active session is deleted, a replacement session is automatically created.

## Auto-Title Policy

- New sessions start with default title `New chat`.
- Automatic title generation runs only when enough dialog context exists.
- Auto-title has retry throttling and a max attempt count to avoid repeated noise.
- Manual rename locks title policy to prevent unwanted retitles.

## Operational Tips

- Keep frequently used sessions pinned.
- Use search before switching when a chat has many sessions.
- Prefer restore from trash over creating a new session after accidental deletion.
- For audits, retain `_session_index.json` and session JSONL files together.
