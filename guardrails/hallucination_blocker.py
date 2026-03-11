"""G1 -- Hallucination blocker for financial data.

Scans LLM output for currency amounts, account numbers, and percentages.
If financial data is detected but no tool call produced it (i.e. it is
not backed by a fact in the fact store), the response is BLOCKED and
replaced with a safe fallback message.
"""

from __future__ import annotations

import re

from memory.fact_store import FactStore

# ── patterns that catch financial data in free text ───────────────

_CURRENCY_AMOUNT = re.compile(
    r"""
    (?:BDT|USD|EUR|GBP|INR|\$|£|€|৳)       # currency symbol / code
    \s*[\d,]+(?:\.\d+)?                      # amount
    |
    [\d,]+(?:\.\d+)?                         # amount first …
    \s*(?:BDT|USD|EUR|GBP|INR|taka|tk)      # … then currency word / code
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ACCOUNT_NUMBER = re.compile(
    r"\b\d{10,16}\b",  # 10-to-16-digit sequences
)

_PERCENTAGE = re.compile(
    r"\b\d+(?:\.\d+)?\s*%",
)

BLOCKED_MESSAGE = "I need to look that up to give you accurate information."


class HallucinationBlocker:
    """G1: block LLM-fabricated financial data."""

    def __init__(self, fact_store: FactStore) -> None:
        self._fact_store = fact_store

    # ── public API ────────────────────────────────────────────────

    def check(self, response: str, had_tool_call: bool) -> tuple[bool, str]:
        """Return ``(passed, reason)``.

        *passed* is ``True`` when the response is safe to send.
        When *passed* is ``False``, *reason* describes why.
        """
        if had_tool_call:
            # Tool results are trusted -- skip scan.
            return True, ""

        # Collect every suspicious financial snippet.
        findings: list[str] = []

        for match in _CURRENCY_AMOUNT.finditer(response):
            value = match.group().strip()
            if not self._value_in_facts(value):
                findings.append(f"currency amount: {value}")

        for match in _ACCOUNT_NUMBER.finditer(response):
            value = match.group().strip()
            if not self._value_in_facts(value):
                findings.append(f"account number: {value}")

        for match in _PERCENTAGE.finditer(response):
            value = match.group().strip()
            if not self._value_in_facts(value):
                findings.append(f"percentage: {value}")

        if findings:
            reason = "Unverified financial data detected: " + "; ".join(findings)
            return False, reason

        return True, ""

    # ── internals ─────────────────────────────────────────────────

    def _value_in_facts(self, value: str) -> bool:
        """Check whether *value* appears in any recorded fact."""
        for fact in self._fact_store.all_latest():
            if value in fact.value:
                return True
        return False
