"""Execute validated tool calls and write results into the fact store.

Design constraints (from the architecture doc):
  - G3 enforcement: every write to fact_store is gated by this executor,
    never by the LLM directly.
  - Validation happens via JSON Schema *before* any mock function runs.
  - On validation failure, a structured error is returned so the agent
    can re-prompt with a corrected call.
"""

from __future__ import annotations

import json
import logging
import time
from types import ModuleType
from typing import Any

import jsonschema

from tools.tool_registry import TOOL_REGISTRY, ToolDefinition

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Validate, execute, and record tool calls."""

    def __init__(
        self,
        registry: dict[str, ToolDefinition] | None = None,
        mock_tools_module: ModuleType | None = None,
    ) -> None:
        self._registry = registry or TOOL_REGISTRY

        if mock_tools_module is None:
            from tools import mock_tools as _mock
            self._tools_module = _mock
        else:
            self._tools_module = mock_tools_module

    # ── validation ─────────────────────────────────────────────────

    def validate_call(
        self,
        function_name: str,
        params: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Check that *function_name* exists and *params* match its schema.

        Returns:
            (True, None) on success, or (False, error_message) on failure.
        """
        tool_def = self._registry.get(function_name)
        if tool_def is None:
            available = ", ".join(sorted(self._registry))
            return False, (
                f"Unknown tool '{function_name}'. "
                f"Available tools: {available}"
            )

        try:
            jsonschema.validate(instance=params, schema=tool_def.params_schema)
        except jsonschema.ValidationError as exc:
            return False, f"Parameter validation failed for '{function_name}': {exc.message}"

        return True, None

    # ── single call execution ──────────────────────────────────────

    def _execute_one(
        self,
        function_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a single validated tool call and return the result envelope."""
        fn = getattr(self._tools_module, function_name, None)
        if fn is None:
            return {
                "tool": function_name,
                "success": False,
                "error": f"Tool '{function_name}' has no implementation.",
            }

        try:
            result = fn(**params)
            return {
                "tool": function_name,
                "success": True,
                "result": result,
            }
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", function_name)
            return {
                "tool": function_name,
                "success": False,
                "error": str(exc),
            }

    # ── batch execution with G3 fact-store writes ──────────────────

    def execute(
        self,
        tool_calls: list[dict[str, Any]],
        fact_store: dict[str, Any] | None = None,
        turn_number: int = 0,
    ) -> list[dict[str, Any]]:
        """Execute a batch of tool calls.

        Each element of *tool_calls* must have at minimum::

            {"function": "tool_name", "params": { ... }}

        After each successful call the result is written into
        *fact_store* (keyed by ``"tool:<name>:<turn>"``), enforcing
        G3: the LLM never writes facts directly.

        Returns a list of result envelopes (same order as input).
        """
        if fact_store is None:
            fact_store = {}

        results: list[dict[str, Any]] = []

        for call in tool_calls:
            function_name: str = call.get("function", "")
            params: dict[str, Any] = call.get("params", {})

            # ── validate ───────────────────────────────────────────
            valid, error_msg = self.validate_call(function_name, params)

            if not valid:
                logger.warning(
                    "Validation failed for %s: %s", function_name, error_msg,
                )
                results.append({
                    "tool": function_name,
                    "success": False,
                    "error": error_msg,
                    "template_fallback": self._fallback_message(
                        function_name, error_msg,
                    ),
                })
                continue

            # ── execute ────────────────────────────────────────────
            envelope = self._execute_one(function_name, params)
            results.append(envelope)

            # ── G3: write to fact store ────────────────────────────
            if envelope["success"] and fact_store is not None:
                result_data = envelope["result"]
                if isinstance(result_data, dict):
                    for k, v in result_data.items():
                        if k not in ("success", "error"):
                            fact_store.add(
                                key=k,
                                value=str(v),
                                turn_number=turn_number,
                                source_tool=function_name,
                            )
                else:
                    fact_store.add(
                        key=f"{function_name}_result",
                        value=str(result_data),
                        turn_number=turn_number,
                        source_tool=function_name,
                    )
                logger.debug("Facts stored for tool: %s", function_name)

        return results

    # ── template fallback ──────────────────────────────────────────

    @staticmethod
    def _fallback_message(function_name: str, error_msg: str) -> str:
        """Produce a user-friendly message when a tool call fails validation."""
        return (
            f"I tried to use the '{function_name}' tool but the request "
            f"was malformed. Error: {error_msg}. "
            f"Let me try again with the correct parameters."
        )
