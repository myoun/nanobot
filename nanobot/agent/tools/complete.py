"""Completion tool for explicit turn finalization."""

import json
from typing import Any

from nanobot.agent.tools.base import Tool


class CompleteTaskTool(Tool):
    """Explicitly mark a turn as completed with a final user-facing answer."""

    @property
    def name(self) -> str:
        return "complete_task"

    @property
    def description(self) -> str:
        return (
            "Mark the current user request as complete. "
            "Use this when you have finished all needed tool work and are ready to deliver "
            "the final response."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "final_answer": {
                    "type": "string",
                    "description": "Final user-facing response for this turn.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Optional confidence score between 0 and 1.",
                },
                "remaining_risks": {
                    "type": "string",
                    "description": "Optional short note about residual uncertainty or caveats.",
                },
            },
            "required": ["final_answer"],
        }

    async def execute(
        self,
        final_answer: str,
        confidence: float | None = None,
        remaining_risks: str | None = None,
        **kwargs: Any,
    ) -> str:
        payload: dict[str, Any] = {"status": "completed", "final_answer": final_answer}
        if confidence is not None:
            payload["confidence"] = confidence
        if remaining_risks:
            payload["remaining_risks"] = remaining_risks
        return json.dumps(payload, ensure_ascii=False)
