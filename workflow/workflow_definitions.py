"""Declarative workflow definitions for multi-step processes.

Each workflow is a list of steps.  Step types:

  - **needs_input** -- pause and ask the user for a value
  - **auto**        -- execute a tool call automatically
  - **proactive**   -- present a confirmation / summary to the user
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowStep:
    """A single step in a workflow."""

    id: str
    type: str  # "needs_input" | "auto" | "proactive"
    prompt: str = ""
    tool_call: dict[str, Any] | None = None
    slot: str = ""


@dataclass(frozen=True)
class WorkflowDefinition:
    """Complete definition of a multi-step workflow."""

    workflow_id: str
    description: str
    steps: tuple[WorkflowStep, ...]


# ── card_block workflow ───────────────────────────────────────────

CARD_BLOCK = WorkflowDefinition(
    workflow_id="card_block",
    description="Block a debit or credit card.",
    steps=(
        WorkflowStep(
            id="ask_card_type",
            type="needs_input",
            prompt="Which card would you like to block -- debit or credit?",
            slot="card_type",
        ),
        WorkflowStep(
            id="ask_reason",
            type="needs_input",
            prompt="What is the reason for blocking? (e.g. lost, stolen, fraud)",
            slot="reason",
        ),
        WorkflowStep(
            id="execute_block",
            type="auto",
            tool_call={
                "function": "block_card",
                "params_from_slots": ["card_type", "reason"],
            },
            slot="block_result",
        ),
        WorkflowStep(
            id="confirm",
            type="proactive",
            prompt=(
                "Your {card_type} card has been blocked successfully. "
                "Reference number: {block_result}. "
                "Is there anything else I can help with?"
            ),
        ),
    ),
)

# ── dispute_flow workflow ─────────────────────────────────────────

DISPUTE_FLOW = WorkflowDefinition(
    workflow_id="dispute_flow",
    description="File a dispute against a transaction.",
    steps=(
        WorkflowStep(
            id="ask_transaction_id",
            type="needs_input",
            prompt="Please provide the transaction ID you'd like to dispute.",
            slot="transaction_id",
        ),
        WorkflowStep(
            id="ask_reason",
            type="needs_input",
            prompt="What is the reason for the dispute?",
            slot="reason",
        ),
        WorkflowStep(
            id="execute_dispute",
            type="auto",
            tool_call={
                "function": "file_dispute",
                "params_from_slots": ["transaction_id", "reason"],
            },
            slot="dispute_result",
        ),
        WorkflowStep(
            id="confirm",
            type="proactive",
            prompt=(
                "Your dispute has been filed successfully. "
                "Dispute ID: {dispute_result}. "
                "Is there anything else I can help with?"
            ),
        ),
    ),
)

# ── registry ──────────────────────────────────────────────────────

WORKFLOW_REGISTRY: dict[str, WorkflowDefinition] = {
    CARD_BLOCK.workflow_id: CARD_BLOCK,
    DISPUTE_FLOW.workflow_id: DISPUTE_FLOW,
}
