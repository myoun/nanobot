# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec - Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated when too long
- `restrictToWorkspace` config can limit file access to the workspace

## cron - Scheduled Reminders

- Use `nanobot cron` commands for one-time or recurring reminders
- Use `--deliver --to USER_ID --channel CHANNEL` to deliver to chat channels

