"""ActionRouter -- the main turn-level orchestrator (D2).

Wires together Pass 1 (Qwen3 routing), Pass 2 (Llama naturalization),
FAQ search, tool execution, workflow engine, guardrails, and the state
machine.  Exposes a single ``process_turn`` method that drives one
complete user -> assistant exchange.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from engine.pass1_resolver import Pass1Resolver, Pass1Result
from engine.pass2_naturalizer import Pass2Naturalizer
from memory.session_store import SessionStore
from memory.fact_store import FactStore
from memory.thread_tracker import ThreadTracker
from memory.emotion_tracker import EmotionTracker
from memory.turn_history import TurnHistory

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Fix 1: action-to-trigger mapping for the state machine
# ------------------------------------------------------------------
_ACTION_TO_TRIGGER = {
    "faq": "policy_question",
    "tool_call": "needs_data",
    "talk": "chit_chat",
    "workflow": "multi_step",
    "escalate": "demands_agent",
}


# ------------------------------------------------------------------
# Lightweight protocols for pluggable subsystems so that the router
# does not depend on concrete implementations at import time.
# ------------------------------------------------------------------
class FAQEngine(Protocol):
    def search(self, query: str) -> Any: ...


class ToolExecutor(Protocol):
    def execute(
        self,
        tool_calls: list[dict[str, Any]],
        fact_store: Any,
        turn_number: int,
    ) -> list[dict[str, Any]]: ...


class WorkflowEngine(Protocol):
    def handle(self, command: dict[str, Any], turn_number: int) -> dict[str, Any]: ...


class Guardrails(Protocol):
    def run_output_checks(
        self,
        response: str,
        had_tool_call: bool,
        hedged: bool,
        faq_sourced: bool,
    ) -> str: ...


class StateMachine(Protocol):
    def transition(self, trigger: str) -> Any: ...
    def can_transition(self, trigger: str) -> bool: ...


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------
@dataclass
class TurnResult:
    """Value object returned by ``ActionRouter.process_turn``."""

    filler_message: str | None
    final_response: str
    action_type: str
    debug: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------
class ActionRouter:
    """Orchestrates a single conversational turn through the pipeline."""

    def __init__(
        self,
        pass1: Pass1Resolver,
        pass2: Pass2Naturalizer,
        faq_engine: FAQEngine,
        tool_executor: ToolExecutor,
        workflow_engine: WorkflowEngine,
        guardrails: Guardrails,
        state_machine: StateMachine,
        session_store: SessionStore,
        fact_store: FactStore,
        thread_tracker: ThreadTracker,
        emotion_tracker: EmotionTracker,
        turn_history: TurnHistory,
    ) -> None:
        self._pass1 = pass1
        self._pass2 = pass2
        self._faq = faq_engine
        self._tools = tool_executor
        self._workflow = workflow_engine
        self._guard = guardrails
        self._sm = state_machine

        self._session = session_store
        self._facts = fact_store
        self._threads = thread_tracker
        self._emotion = emotion_tracker
        self._turns = turn_history

        self._turn_counter: int = len(turn_history)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def process_turn(self, utterance: str) -> TurnResult:
        """Run the full two-pass pipeline for one user utterance."""

        self._turn_counter += 1
        turn_num = self._turn_counter

        # 1. Record user utterance
        self._turns.add(turn_number=turn_num, role="user", text=utterance)

        # 2. Pass 1: agentic routing via Qwen3
        p1: Pass1Result = self._pass1.resolve(
            self._session,
            self._facts,
            self._threads,
            self._emotion,
            self._turns,
            utterance,
        )

        # 3. Update emotion tracker (tolerate unknown labels gracefully)
        try:
            self._emotion.record(turn_num, p1.emotion_read)
        except ValueError:
            log.warning("Emotion '%s' rejected; recording neutral.", p1.emotion_read)
            self._emotion.record(turn_num, "neutral")

        # 4. Route on action_type
        response: str
        debug: dict[str, Any] = {"pass1": p1.raw_json}
        faq_sourced: bool = False
        had_tool_call: bool = p1.action_type == "tool_call"
        hedged: bool = False
        tool_calls_list: list[dict[str, Any]] | None = None
        tool_results_list: list[dict[str, Any]] | None = None

        # Fix 1: State-machine transition using _ACTION_TO_TRIGGER mapping
        action_trigger = _ACTION_TO_TRIGGER.get(p1.action_type)
        if action_trigger:
            try:
                if self._sm.can_transition(action_trigger):
                    self._sm.transition(action_trigger)
            except Exception:
                log.warning(
                    "State-machine transition failed for trigger '%s'.",
                    action_trigger,
                )

        try:
            response, debug, faq_sourced, hedged, tool_calls_list, tool_results_list = (
                self._route(p1, turn_num, debug, utterance)
            )
        except Exception:
            log.exception("Error during action routing; producing safe fallback.")
            response = (
                "I ran into a temporary issue while looking that up. "
                "Could you please try again in a moment?"
            )
            debug["routing_error"] = True

        # 5. Guardrails (Fix 8: pass faq_sourced flag)
        response = self._guard.run_output_checks(
            response,
            had_tool_call=had_tool_call,
            hedged=hedged,
            faq_sourced=faq_sourced,
        )

        # 6. Record assistant response (Fix 5: include tool_calls and tool_results)
        self._turns.add(
            turn_number=turn_num,
            role="assistant",
            text=response,
            tool_calls=tool_calls_list,
            tool_results=tool_results_list,
        )

        # 7. Send "answered" trigger to return state machine to READY
        try:
            if self._sm.can_transition("answered"):
                self._sm.transition("answered")
        except Exception:
            log.warning("State-machine 'answered' transition failed.")

        return TurnResult(
            filler_message=p1.filler_message,
            final_response=response,
            action_type=p1.action_type,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # internal routing
    # ------------------------------------------------------------------
    def _route(
        self,
        p1: Pass1Result,
        turn_num: int,
        debug: dict[str, Any],
        utterance: str,
    ) -> tuple[str, dict[str, Any], bool, bool, list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        """Dispatch to the correct handler based on ``action_type``.

        Returns:
            (response, debug, faq_sourced, hedged, tool_calls, tool_results)
        """

        action = p1.action_type

        if action == "faq":
            return self._handle_faq(p1, debug, utterance)

        if action == "tool_call":
            return self._handle_tool_call(p1, turn_num, debug, utterance)

        if action == "talk":
            return self._handle_talk(p1, debug)

        if action == "workflow":
            return self._handle_workflow(p1, turn_num, debug)

        if action == "escalate":
            return self._handle_escalate(p1, debug)

        # Unknown action -- treat as talk
        log.warning("Unknown action_type '%s'; treating as talk.", action)
        return self._handle_talk(p1, debug)

    # ---- FAQ -----------------------------------------------------------
    def _handle_faq(
        self,
        p1: Pass1Result,
        debug: dict[str, Any],
        utterance: str,
    ) -> tuple[str, dict[str, Any], bool, bool, None, None]:
        # Fix 3: resolved_query fallback -- use raw utterance if no resolved_query
        query = p1.resolved_query or utterance
        faq_result = self._faq.search(query)

        debug["faq_result"] = str(faq_result)
        debug["faq_tier"] = faq_result.tier
        debug["faq_confidence"] = faq_result.confidence
        debug["faq_hedged"] = faq_result.hedged

        # Fix 4: Thread tracker -- open thread with topic from resolved query
        topic = p1.resolved_query or utterance
        try:
            self._threads.open(topic)
        except Exception:
            log.warning("Failed to open thread for FAQ topic '%s'.", topic)

        # Fix 2: FAQ tier handling per D6
        if faq_result.tier == "PURE_FAQ" and faq_result.confidence > 0.85:
            # PURE_FAQ: use answer directly -- skip Pass 2 LLM call entirely
            response = faq_result.answer or ""
            debug["faq_pass2_skipped"] = True

            # Resolve the thread immediately -- FAQ fully answered
            try:
                self._threads.resolve(topic)
            except Exception:
                log.warning("Failed to resolve thread for FAQ topic '%s'.", topic)

            return response, debug, True, False, None, None

        if faq_result.tier == "BLENDED":
            # BLENDED (0.6-0.85): pass answer as ground_truth + hedged=True
            # Fix 7: pass faq_result.answer as ground_truth string, not dataclass
            response = self._pass2.naturalize(
                pass1_result=p1,
                ground_truth=faq_result.answer,
                session_store=self._session,
                emotion_state=p1.emotion_read,
                hedged=True,
            )

            # Resolve the thread -- FAQ answered (with hedging)
            try:
                self._threads.resolve(topic)
            except Exception:
                log.warning("Failed to resolve thread for FAQ topic '%s'.", topic)

            return response, debug, True, True, None, None

        # PURE_LLM (<0.6): pass ground_truth=None, let LLM generate freely
        response = self._pass2.naturalize(
            pass1_result=p1,
            ground_truth=None,
            session_store=self._session,
            emotion_state=p1.emotion_read,
            hedged=False,
        )

        # Resolve the thread -- LLM answered freely
        try:
            self._threads.resolve(topic)
        except Exception:
            log.warning("Failed to resolve thread for FAQ topic '%s'.", topic)

        return response, debug, True, False, None, None

    # ---- Tool call -----------------------------------------------------
    def _handle_tool_call(
        self,
        p1: Pass1Result,
        turn_num: int,
        debug: dict[str, Any],
        utterance: str,
    ) -> tuple[str, dict[str, Any], bool, bool, list[dict[str, Any]], list[dict[str, Any]]]:
        tool_calls = p1.tool_calls or []
        results_list = self._tools.execute(
            tool_calls, fact_store=self._facts, turn_number=turn_num,
        )
        debug["tool_result"] = results_list

        # Fix 4: Thread tracker -- open/update thread, resolve after success
        topic = p1.resolved_query or utterance
        try:
            self._threads.open(topic)
        except Exception:
            log.warning("Failed to open thread for tool topic '%s'.", topic)

        # Build ground truth from all tool results
        ground_truth_parts = []
        all_succeeded = True
        for r in results_list:
            if r.get("success"):
                ground_truth_parts.append(str(r.get("result", "")))
            else:
                all_succeeded = False
        ground_truth = "; ".join(ground_truth_parts) if ground_truth_parts else "Tool execution failed."
        emotion_state = p1.emotion_read

        response = self._pass2.naturalize(
            pass1_result=p1,
            ground_truth=ground_truth,
            session_store=self._session,
            emotion_state=emotion_state,
            hedged=False,
        )

        # Fix 4: Resolve thread after successful tool result
        if all_succeeded and ground_truth_parts:
            try:
                self._threads.resolve(topic)
            except Exception:
                log.warning("Failed to resolve thread for tool topic '%s'.", topic)

        # Fix 5: return tool_calls and tool_results for turn history
        return response, debug, False, False, tool_calls, results_list

    # ---- Talk (chit-chat) ----------------------------------------------
    def _handle_talk(
        self,
        p1: Pass1Result,
        debug: dict[str, Any],
    ) -> tuple[str, dict[str, Any], bool, bool, None, None]:
        response = self._pass2.naturalize(
            pass1_result=p1,
            ground_truth=None,
            session_store=self._session,
            emotion_state=p1.emotion_read,
        )
        debug["talk_subtype"] = p1.talk_subtype
        return response, debug, False, False, None, None

    # ---- Workflow -------------------------------------------------------
    def _handle_workflow(
        self,
        p1: Pass1Result,
        turn_num: int,
        debug: dict[str, Any],
    ) -> tuple[str, dict[str, Any], bool, bool, None, None]:
        command = p1.workflow_command or {}
        # Fix 6: Pass turn_num to workflow_engine.handle()
        wf_result = self._workflow.handle(command, turn_number=turn_num)
        debug["workflow_result"] = wf_result

        ground_truth = wf_result.get("message", str(wf_result))
        emotion_state = p1.emotion_read

        response = self._pass2.naturalize(
            pass1_result=p1,
            ground_truth=ground_truth,
            session_store=self._session,
            emotion_state=emotion_state,
        )
        return response, debug, False, False, None, None

    # ---- Escalate -------------------------------------------------------
    def _handle_escalate(
        self,
        p1: Pass1Result,
        debug: dict[str, Any],
    ) -> tuple[str, dict[str, Any], bool, bool, None, None]:
        debug["escalation"] = True

        # Fix 4: Suspend all threads on escalation
        for thread in self._threads.all_threads:
            if thread.status.value == "ACTIVE":
                try:
                    self._threads.suspend(thread.topic)
                except Exception:
                    log.warning(
                        "Failed to suspend thread '%s' during escalation.",
                        thread.topic,
                    )

        response = (
            "I understand this is important to you, and I want to make sure "
            "you get the best help possible. Let me connect you with a "
            "specialist who can assist you further. Please hold on for "
            "just a moment."
        )
        return response, debug, False, False, None, None
