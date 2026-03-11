"""Multi-step workflow engine.

Manages one or more concurrent workflows, each advancing through their
declared steps.  The engine keeps track of collected slot values and
controls step transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import WorkflowConfig
from memory.fact_store import FactStore
from workflow.workflow_definitions import (
    WORKFLOW_REGISTRY,
    WorkflowDefinition,
    WorkflowStep,
)

logger = logging.getLogger(__name__)

_CFG = WorkflowConfig()


# ── state ─────────────────────────────────────────────────────────

@dataclass
class WorkflowState:
    """Runtime state of a single active workflow."""

    workflow_id: str
    current_step_index: int = 0
    slots: dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # "active" | "suspended" | "completed" | "expired"
    started_turn: int = 0
    last_active_turn: int = 0


# ── engine ────────────────────────────────────────────────────────

class WorkflowEngine:
    """Orchestrate multi-step workflows."""

    def __init__(self) -> None:
        self._active: dict[str, WorkflowState] = {}

    # ── public API ────────────────────────────────────────────────

    def handle(self, command: dict[str, Any], turn_number: int) -> dict[str, Any]:
        """Dispatch a workflow command (start / resume / pause).

        *command* must contain at least ``{"action": "...", "workflow_id": "..."}``.
        """
        action = command.get("action", "")
        workflow_id = command.get("workflow_id", "")

        if action == "start":
            return self._start(workflow_id, turn_number)
        elif action == "resume":
            user_input = command.get("user_input", "")
            fact_store = command.get("fact_store")
            return self.advance_step(workflow_id, user_input, turn_number, fact_store)
        elif action == "pause":
            self.suspend(workflow_id, turn_number)
            return {
                "workflow_id": workflow_id,
                "status": "suspended",
                "message": "Workflow paused. You can resume anytime.",
            }
        else:
            return {"error": f"Unknown workflow action '{action}'."}

    def advance_step(
        self,
        workflow_id: str,
        user_input: str,
        turn_number: int,
        fact_store: FactStore | None = None,
    ) -> dict[str, Any]:
        """Store the user's input for the current step and advance.

        Returns a dict describing what the caller should do next:
          - ``{"type": "needs_input", "prompt": "..."}``
          - ``{"type": "auto", "tool_call": {...}}``
          - ``{"type": "proactive", "message": "..."}``
          - ``{"type": "completed", "message": "..."}``
        """
        state = self._active.get(workflow_id)
        if state is None:
            return {"error": f"No active workflow '{workflow_id}'."}

        if state.status != "active":
            return {"error": f"Workflow '{workflow_id}' is {state.status}."}

        definition = WORKFLOW_REGISTRY.get(workflow_id)
        if definition is None:
            return {"error": f"Unknown workflow definition '{workflow_id}'."}

        # Store input for the current step's slot (if applicable).
        current_step = definition.steps[state.current_step_index]
        if current_step.slot and user_input:
            state.slots[current_step.slot] = user_input

            # Write to fact store if available.
            if fact_store is not None:
                fact_store.add(
                    key=f"workflow:{workflow_id}:{current_step.slot}",
                    value=user_input,
                    turn_number=turn_number,
                    source_tool="workflow_engine",
                )

        state.last_active_turn = turn_number

        # Move to the next step.
        state.current_step_index += 1

        if state.current_step_index >= len(definition.steps):
            state.status = "completed"
            return {
                "workflow_id": workflow_id,
                "type": "completed",
                "message": "Workflow completed.",
                "slots": dict(state.slots),
            }

        return self._step_response(definition, state)

    def suspend(self, workflow_id: str, turn_number: int) -> None:
        """Pause a workflow so it can be resumed later."""
        state = self._active.get(workflow_id)
        if state is not None:
            state.status = "suspended"
            state.last_active_turn = turn_number
            logger.info("Workflow '%s' suspended at turn %d", workflow_id, turn_number)

    def check_expiry(self, current_turn: int, max_idle: int | None = None) -> list[str]:
        """Return IDs of workflows that have been idle too long."""
        if max_idle is None:
            max_idle = _CFG.MAX_IDLE_TURNS

        expired: list[str] = []
        for wf_id, state in self._active.items():
            if state.status == "active":
                idle = current_turn - state.last_active_turn
                if idle > max_idle:
                    state.status = "expired"
                    expired.append(wf_id)
                    logger.info(
                        "Workflow '%s' expired after %d idle turns",
                        wf_id, idle,
                    )

        return expired

    # ── properties ────────────────────────────────────────────────

    @property
    def active_workflows(self) -> dict[str, WorkflowState]:
        return dict(self._active)

    def get_state(self, workflow_id: str) -> WorkflowState | None:
        return self._active.get(workflow_id)

    # ── internals ─────────────────────────────────────────────────

    def _start(self, workflow_id: str, turn_number: int) -> dict[str, Any]:
        """Initialise a new workflow and return the first step."""
        definition = WORKFLOW_REGISTRY.get(workflow_id)
        if definition is None:
            return {"error": f"Unknown workflow '{workflow_id}'."}

        if workflow_id in self._active:
            existing = self._active[workflow_id]
            if existing.status == "active":
                return {
                    "error": f"Workflow '{workflow_id}' is already active.",
                }
            # Allow restarting a completed / expired / suspended workflow.

        state = WorkflowState(
            workflow_id=workflow_id,
            started_turn=turn_number,
            last_active_turn=turn_number,
        )
        self._active[workflow_id] = state

        logger.info("Workflow '%s' started at turn %d", workflow_id, turn_number)
        return self._step_response(definition, state)

    def _step_response(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """Build the response dict for the current step."""
        step = definition.steps[state.current_step_index]

        if step.type == "needs_input":
            return {
                "workflow_id": definition.workflow_id,
                "type": "needs_input",
                "prompt": step.prompt,
                "slot": step.slot,
                "step_id": step.id,
            }

        if step.type == "auto":
            # Build params from collected slots.
            tool_call = dict(step.tool_call) if step.tool_call else {}
            params_from_slots = tool_call.pop("params_from_slots", [])
            params = {slot: state.slots.get(slot, "") for slot in params_from_slots}
            return {
                "workflow_id": definition.workflow_id,
                "type": "auto",
                "tool_call": {
                    "function": tool_call.get("function", ""),
                    "params": params,
                },
                "slot": step.slot,
                "step_id": step.id,
            }

        if step.type == "proactive":
            # Format the prompt template with collected slot values.
            try:
                message = step.prompt.format(**state.slots)
            except KeyError:
                message = step.prompt
            return {
                "workflow_id": definition.workflow_id,
                "type": "proactive",
                "message": message,
                "step_id": step.id,
            }

        return {"error": f"Unknown step type '{step.type}'."}
