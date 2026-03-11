"""G2 -- Tool-call parameter validator.

Thin wrapper around jsonschema validation.  The heavy lifting is done
inside ``ToolExecutor.validate_call``; this module provides a standalone
entry point that can be invoked before dispatch (e.g. from the guardrails
pipeline) when an early rejection is desired.
"""

from __future__ import annotations

from typing import Any

import jsonschema

from tools.tool_registry import TOOL_REGISTRY, ToolDefinition


def validate(
    function_name: str,
    params: dict[str, Any],
    registry: dict[str, ToolDefinition] | None = None,
) -> tuple[bool, str]:
    """Validate *params* against the JSON Schema of *function_name*.

    Returns:
        ``(True, "")`` when valid, or ``(False, error_message)`` when not.
    """
    reg = registry or TOOL_REGISTRY

    tool_def = reg.get(function_name)
    if tool_def is None:
        available = ", ".join(sorted(reg))
        return False, (
            f"Unknown tool '{function_name}'. Available tools: {available}"
        )

    try:
        jsonschema.validate(instance=params, schema=tool_def.params_schema)
    except jsonschema.ValidationError as exc:
        return False, (
            f"Parameter validation failed for '{function_name}': {exc.message}"
        )

    return True, ""
