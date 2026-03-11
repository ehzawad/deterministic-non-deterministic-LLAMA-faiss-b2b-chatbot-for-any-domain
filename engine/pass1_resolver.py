"""Pass1Resolver -- Qwen3 structured-output pass.

Sends the full agentic prompt to Qwen3-4B, parses the JSON response,
validates required fields, and returns a typed ``Pass1Result``.  Includes
one automatic retry on parse failure before falling back to a safe
``talk`` action.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from engine.llm import DualLLMEngine
from engine.context_builder import ContextBuilder
from memory.session_store import SessionStore
from memory.fact_store import FactStore
from memory.thread_tracker import ThreadTracker
from memory.emotion_tracker import EmotionTracker
from memory.turn_history import TurnHistory

log = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"faq", "tool_call", "talk", "workflow", "escalate"})
_VALID_TALK_SUBTYPES = frozenset(
    {"greeting", "empathy", "clarification", "proactive", None}
)
_VALID_EMOTIONS = frozenset(
    {"neutral", "concerned", "frustrated", "anxious", "angry", "stressed"}
)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------
@dataclass
class Pass1Result:
    """Typed representation of the Qwen3 agent output."""

    action_type: str  # faq | tool_call | talk | workflow | escalate
    resolved_query: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    workflow_command: dict[str, Any] | None = None
    talk_subtype: str | None = None  # greeting | empathy | clarification | proactive
    direct_response: str | None = None
    emotion_read: str = "neutral"
    filler_message: str | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Resolver
# ------------------------------------------------------------------
class Pass1Resolver:
    """Runs Qwen3-4B to produce a structured routing decision."""

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
    def resolve(
        self,
        session_store: SessionStore,
        fact_store: FactStore,
        thread_tracker: ThreadTracker,
        emotion_tracker: EmotionTracker,
        turn_history: TurnHistory,
        current_utterance: str,
    ) -> Pass1Result:
        """Build the Pass-1 prompt, call Qwen3, parse, validate, return."""
        system_prompt, user_message = self._ctx.build_pass1_prompt(
            session_store,
            fact_store,
            thread_tracker,
            emotion_tracker,
            turn_history,
            current_utterance,
        )

        raw_text = self._llm.agent_generate(
            system_prompt,
            user_message,
            max_tokens=512,
            temperature=0.6,
        )

        parsed = self._try_parse(raw_text)
        if parsed is not None:
            return self._build_result(parsed, raw_text)

        # ---- retry once ----
        log.warning("Pass 1 JSON parse failed; retrying once.")
        raw_text = self._llm.agent_generate(
            system_prompt,
            user_message,
            max_tokens=512,
            temperature=0.3,
        )
        parsed = self._try_parse(raw_text)
        if parsed is not None:
            return self._build_result(parsed, raw_text)

        # ---- fallback: safe talk action ----
        log.error("Pass 1 JSON parse failed after retry; falling back to talk.")
        return Pass1Result(
            action_type="talk",
            talk_subtype="clarification",
            direct_response=(
                "I'm sorry, I didn't quite catch that. Could you rephrase?"
            ),
            emotion_read="neutral",
            raw_json={"_fallback": True, "_raw": raw_text},
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove Qwen3 ``<think>...</think>`` blocks if present."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Find the first top-level ``{...}`` in *text*."""
        depth = 0
        start: int | None = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]
        return None

    def _try_parse(self, raw_text: str) -> dict[str, Any] | None:
        """Attempt to parse raw model output into a dict."""
        cleaned = self._strip_thinking_tags(raw_text)
        json_str = self._extract_json(cleaned)
        if json_str is None:
            return None
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None
            return data
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _build_result(data: dict[str, Any], raw_text: str) -> Pass1Result:
        """Convert parsed dict into a validated ``Pass1Result``."""
        action_type = data.get("action_type", "talk")
        if action_type not in _VALID_ACTIONS:
            log.warning("Unknown action_type '%s'; defaulting to talk.", action_type)
            action_type = "talk"

        emotion = data.get("emotion_read", "neutral")
        if emotion not in _VALID_EMOTIONS:
            log.warning("Unknown emotion '%s'; defaulting to neutral.", emotion)
            emotion = "neutral"

        talk_sub = data.get("talk_subtype")
        if talk_sub not in _VALID_TALK_SUBTYPES:
            talk_sub = None

        tool_calls = data.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            tool_calls = None

        workflow_cmd = data.get("workflow_command")
        if workflow_cmd is not None and not isinstance(workflow_cmd, dict):
            workflow_cmd = None

        return Pass1Result(
            action_type=action_type,
            resolved_query=data.get("resolved_query"),
            tool_calls=tool_calls,
            workflow_command=workflow_cmd,
            talk_subtype=talk_sub,
            direct_response=data.get("direct_response"),
            emotion_read=emotion,
            filler_message=data.get("filler_message"),
            raw_json=data,
        )
