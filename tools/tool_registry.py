"""Tool definitions with JSON-Schema parameter validation.

Every tool the LLM may call is declared here once.  The schema is
re-used for both prompt injection (so the agent knows the signature)
and runtime validation (so bad calls are caught before execution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable descriptor for a single callable tool."""

    name: str
    description: str
    params_schema: dict[str, Any]
    returns: str


# ── tool definitions ───────────────────────────────────────────────

TOOL_REGISTRY: dict[str, ToolDefinition] = {}


def _register(tool: ToolDefinition) -> None:
    TOOL_REGISTRY[tool.name] = tool


# 1. verify_phone ───────────────────────────────────────────────────
_register(ToolDefinition(
    name="verify_phone",
    description="Look up a customer by phone number and verify identity.",
    params_schema={
        "type": "object",
        "properties": {
            "phone": {
                "type": "string",
                "description": "Bangladeshi phone number (e.g. 01712345678).",
            },
        },
        "required": ["phone"],
        "additionalProperties": False,
    },
    returns="Customer record with name, account_type, tenure, verified flag.",
))

# 2. get_balance ────────────────────────────────────────────────────
_register(ToolDefinition(
    name="get_balance",
    description="Retrieve the current balance for a given account type.",
    params_schema={
        "type": "object",
        "properties": {
            "account_type": {
                "type": "string",
                "enum": ["savings", "current"],
                "description": "Type of bank account.",
            },
        },
        "required": ["account_type"],
        "additionalProperties": False,
    },
    returns="Object with account_type, balance (numeric), and currency.",
))

# 3. block_card ─────────────────────────────────────────────────────
_register(ToolDefinition(
    name="block_card",
    description="Immediately block a debit or credit card.",
    params_schema={
        "type": "object",
        "properties": {
            "card_type": {
                "type": "string",
                "enum": ["debit", "credit"],
                "description": "Type of card to block.",
            },
            "reason": {
                "type": "string",
                "description": "Reason for blocking (e.g. lost, stolen, fraud).",
            },
        },
        "required": ["card_type", "reason"],
        "additionalProperties": False,
    },
    returns="Confirmation with status, blocking reference number, and card_type.",
))

# 4. file_dispute ───────────────────────────────────────────────────
_register(ToolDefinition(
    name="file_dispute",
    description="File a dispute against a specific transaction.",
    params_schema={
        "type": "object",
        "properties": {
            "transaction_id": {
                "type": "string",
                "description": "The unique transaction identifier.",
            },
            "reason": {
                "type": "string",
                "description": "Why the transaction is being disputed.",
            },
        },
        "required": ["transaction_id", "reason"],
        "additionalProperties": False,
    },
    returns="Dispute record with dispute_id, status, and estimated_resolution.",
))

# 5. get_transaction_history ────────────────────────────────────────
_register(ToolDefinition(
    name="get_transaction_history",
    description="Fetch recent transaction history for an account.",
    params_schema={
        "type": "object",
        "properties": {
            "account_type": {
                "type": "string",
                "enum": ["savings", "current"],
                "description": "Type of bank account.",
            },
            "days": {
                "type": "integer",
                "default": 30,
                "description": "Number of past days to include (default 30).",
            },
        },
        "required": ["account_type"],
        "additionalProperties": False,
    },
    returns="List of recent transactions with date, amount, description, and type.",
))

# 6. get_credit_score ──────────────────────────────────────────────
_register(ToolDefinition(
    name="get_credit_score",
    description="Retrieve the customer's current credit score.",
    params_schema={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    returns="Object with numeric score and human-readable rating.",
))
