"""Mock tool implementations returning realistic Bangladeshi banking data.

Every function here mirrors one entry in ``tool_registry.TOOL_REGISTRY``.
The signatures accept **kwargs so the executor can forward validated params
without unpacking by hand.
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from typing import Any


# ── hardcoded customer database ────────────────────────────────────

CUSTOMER_DB: dict[str, dict[str, Any]] = {
    "01712345678": {
        "name": "Rafiqul Islam",
        "account_type": "savings",
        "tenure": "5 years",
        "verified": True,
        "phone": "01712345678",
    },
    "01898765432": {
        "name": "Fatema Begum",
        "account_type": "current",
        "tenure": "3 years",
        "verified": True,
        "phone": "01898765432",
    },
    "01551234567": {
        "name": "Kamal Hossain",
        "account_type": "savings",
        "tenure": "1 year",
        "verified": True,
        "phone": "01551234567",
    },
}


# ── helper ─────────────────────────────────────────────────────────

def _ref(prefix: str, length: int = 5) -> str:
    """Generate a reference code like BLK-A3K9F."""
    chars = string.ascii_uppercase + string.digits
    return f"{prefix}-{''.join(random.choices(chars, k=length))}"


# ── tool implementations ──────────────────────────────────────────

def verify_phone(*, phone: str) -> dict[str, Any]:
    """Look up customer by Bangladeshi mobile number."""
    normalized = phone.strip().replace("-", "").replace(" ", "")
    customer = CUSTOMER_DB.get(normalized)

    if customer is None:
        return {
            "verified": False,
            "error": f"No account found for phone number {phone}.",
        }

    return {
        "verified": customer["verified"],
        "name": customer["name"],
        "account_type": customer["account_type"],
        "tenure": customer["tenure"],
        "phone": customer["phone"],
    }


def get_balance(*, account_type: str) -> dict[str, Any]:
    """Return a mock balance in BDT."""
    balances = {
        "savings": 87_345.50,
        "current": 2_41_620.75,
    }
    return {
        "account_type": account_type,
        "balance": balances.get(account_type, 0.0),
        "currency": "BDT",
    }


def block_card(*, card_type: str, reason: str) -> dict[str, Any]:
    """Block the given card immediately."""
    return {
        "status": "blocked",
        "reference": _ref("BLK"),
        "card_type": card_type,
        "reason": reason,
        "blocked_at": datetime.now().isoformat(timespec="seconds"),
    }


def file_dispute(*, transaction_id: str, reason: str) -> dict[str, Any]:
    """File a dispute against a transaction."""
    return {
        "dispute_id": _ref("DSP", length=4),
        "transaction_id": transaction_id,
        "status": "filed",
        "reason": reason,
        "estimated_resolution": "5-7 business days",
        "filed_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_transaction_history(
    *,
    account_type: str,
    days: int = 30,
) -> dict[str, Any]:
    """Return the five most recent transactions."""
    today = datetime.now()
    transactions = [
        {
            "date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "description": "bKash Transfer to 01812345678",
            "amount": -2_500.00,
            "type": "debit",
            "balance_after": 84_845.50,
        },
        {
            "date": (today - timedelta(days=3)).strftime("%Y-%m-%d"),
            "description": "Salary Credit - ACI Limited",
            "amount": 65_000.00,
            "type": "credit",
            "balance_after": 87_345.50,
        },
        {
            "date": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
            "description": "Utility Bill - DESCO Electricity",
            "amount": -3_450.00,
            "type": "debit",
            "balance_after": 22_345.50,
        },
        {
            "date": (today - timedelta(days=8)).strftime("%Y-%m-%d"),
            "description": "POS Purchase - Agora Supermarket Gulshan",
            "amount": -1_870.25,
            "type": "debit",
            "balance_after": 25_795.50,
        },
        {
            "date": (today - timedelta(days=12)).strftime("%Y-%m-%d"),
            "description": "ATM Withdrawal - Dutch-Bangla Bank Motijheel",
            "amount": -5_000.00,
            "type": "debit",
            "balance_after": 27_665.75,
        },
    ]
    return {
        "account_type": account_type,
        "days_requested": days,
        "transactions": transactions,
        "currency": "BDT",
    }


def get_credit_score() -> dict[str, Any]:
    """Return the customer's credit score."""
    return {
        "score": 742,
        "rating": "Good",
        "scale": "300-900",
        "as_of": datetime.now().strftime("%Y-%m-%d"),
    }
