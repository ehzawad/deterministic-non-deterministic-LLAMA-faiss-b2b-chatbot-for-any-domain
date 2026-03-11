"""Pass2Naturalizer -- Llama 3.1 8B language generation pass.

Takes the structured output from Pass 1 together with ground-truth data
(from tools, FAQ, or workflow) and produces a natural-language customer
response via the Llama generator model.

For simple ``talk`` actions (greetings, empathy) the direct_response
from Pass 1 is returned as-is, skipping a redundant LLM call.
"""

from __future__ import annotations

import logging
from typing import Any

from engine.llm import DualLLMEngine
from engine.context_builder import ContextBuilder
from engine.pass1_resolver import Pass1Result
from memory.session_store import SessionStore

log = logging.getLogger(__name__)


class Pass2Naturalizer:
    """Converts structured data into conversational language via Llama 3.1."""

    def __init__(
        self,
        llm_engine: DualLLMEngine,
        context_builder: ContextBuilder,
    ) -> None:
        self._llm = llm_engine
        self._ctx = context_builder

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def naturalize(
        self,
        pass1_result: Pass1Result,
        ground_truth: str | None,
        faq_result: Any | None = None,
        session_store: SessionStore | None = None,
        emotion_state: str = "neutral",
        hedged: bool = False,
        skip_llm: bool = False,
    ) -> str:
        """Produce a natural-language reply for the customer.

        For ``talk`` actions where ``direct_response`` is already
        populated by the agent, the generator is skipped to save
        latency.

        When *skip_llm* is True (used for PURE_FAQ tier), the
        ground_truth is returned directly without calling Llama.
        """
        # ---- short-circuit: skip_llm for PURE_FAQ tier ----
        if skip_llm and ground_truth:
            log.debug("Skipping Pass 2 -- skip_llm=True, returning ground_truth.")
            return ground_truth

        # ---- short-circuit for simple talk ----
        if (
            pass1_result.action_type == "talk"
            and pass1_result.direct_response
            and ground_truth is None
            and faq_result is None
        ):
            log.debug("Skipping Pass 2 -- using direct_response from Pass 1.")
            return pass1_result.direct_response

        # ---- build prompt ----
        prompt = self._ctx.build_pass2_prompt(
            pass1_result=pass1_result.raw_json,
            ground_truth=ground_truth,
            faq_result=faq_result,
            session_store=session_store,
            emotion_state=emotion_state,
            hedged=hedged,
        )

        # ---- generate ----
        raw = self._llm.generator_generate(
            prompt,
            max_tokens=200,
            temperature=0.7,
            stop=[
                "\n\n\n", "```", "</s>",
                "\nUser:", "\nCustomer:", "\nBot:",
                "\n[System]", "\n[Instruction]", "\n[Tier",
            ],
        )

        response = raw.strip()
        # Strip leading prompt leakage (model sometimes echoes instruction fragments)
        _LEAD_NOISE = [
            "to the question", "to their question", ", as if you",
            ", no chit-chat", ", without introduction", "if possible",
            ", don't say", ", do not introduce", ", no filler",
            "- Keep it", "- Use the tone", "- Make sure",
            "- Be warm",
        ]
        for _ in range(3):  # multiple passes for stacked prefixes
            stripped = False
            for prefix in _LEAD_NOISE:
                if response.lower().startswith(prefix.lower()):
                    response = response[len(prefix):].lstrip("\n ,;:-")
                    stripped = True
                    break
            if not stripped:
                break

        # Clean up trailing LLM artifacts (truncate at first noise marker)
        _TRAIL_NOISE = [
            "```", "---", "Note:", "P.S.",
            "\nUser:", "\nCustomer:", "\nBot:",
            "Please let me know", "Here is the revised",
            "Here is your conversational",
        ]
        for marker in _TRAIL_NOISE:
            if marker in response:
                response = response[:response.index(marker)].strip()
        if not response:
            log.warning(
                "Pass 2 produced empty output; falling back to direct_response."
            )
            return pass1_result.direct_response or (
                "I have the information but had trouble putting it into words. "
                "Let me try again -- could you repeat your question?"
            )

        return response
