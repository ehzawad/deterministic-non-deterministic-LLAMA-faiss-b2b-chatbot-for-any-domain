"""Guardrails pipeline -- runs G1, G4, G5 checks on every outgoing response.

The pipeline is called after the LLM generates an answer and before the
response is sent to the user.  Each guardrail can modify or replace the
response.
"""

from __future__ import annotations

from guardrails.hallucination_blocker import HallucinationBlocker, BLOCKED_MESSAGE
from guardrails.hedge_enforcer import check as hedge_check, fix as hedge_fix
from guardrails.emotion_escalation import check as emotion_check
from memory.fact_store import FactStore
from memory.emotion_tracker import EmotionTracker


class GuardrailsPipeline:
    """Orchestrate output-side guardrails (G1, G4, G5)."""

    def __init__(
        self,
        fact_store: FactStore,
        emotion_tracker: EmotionTracker,
    ) -> None:
        self._fact_store = fact_store
        self._emotion_tracker = emotion_tracker
        self._hallucination_blocker = HallucinationBlocker(fact_store)

    # ── main entry point ──────────────────────────────────────────

    def run_output_checks(
        self,
        response: str,
        had_tool_call: bool,
        hedged: bool,
        faq_sourced: bool = False,
    ) -> str:
        """Run all output guardrails and return the (possibly modified) response.

        Order of checks:
          1. G1  -- hallucination blocker (may replace the entire response)
                    Skipped when *faq_sourced* is True because FAQ answers
                    from our knowledge base are not hallucinations even if
                    they contain financial amounts like "1000 BDT".
          2. G4  -- hedge enforcer (may reword forbidden absolutes)
          3. G5  -- emotion escalation (may append a handoff offer)
        """
        # ── G1: hallucination check ──────────────────────────────
        if not faq_sourced:
            passed, _reason = self._hallucination_blocker.check(response, had_tool_call)
            if not passed:
                return BLOCKED_MESSAGE

        # ── G4: hedge enforcement (only for medium-confidence FAQ) ──
        if hedged:
            hedge_passed, _violations = hedge_check(response, hedged=True)
            if not hedge_passed:
                response = hedge_fix(response)

        # ── G5: emotion escalation ───────────────────────────────
        escalating, escalation_message = emotion_check(self._emotion_tracker)
        if escalating:
            response = response + "\n\n" + escalation_message

        return response
