"""Dialogue state machine from the D8 spec.

Implements a deterministic finite-state machine that tracks where a
conversation is in its lifecycle -- from greeting through authentication,
intent handling (FAQ / tool / workflow), and termination.

Every transition is explicit: if a (state, trigger) pair is not in the
table the machine refuses to move and ``transition()`` raises.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import NamedTuple


# ------------------------------------------------------------------
# states
# ------------------------------------------------------------------

class DialogueState(Enum):
    """Every state the dialogue can occupy."""

    # --- opening / auth ---
    GREETING = auto()
    AUTHENTICATING = auto()
    COLLECT_ID = auto()
    VERIFY = auto()
    LOCKED = auto()
    VERIFIED = auto()

    # --- core conversation ---
    LISTENING = auto()
    READY = auto()
    CLARIFYING = auto()

    # --- FAQ branch ---
    FAQ_LOOKUP = auto()
    FAQ_SEARCHING = auto()
    PURE_FAQ = auto()
    BLENDED_FAQ = auto()
    LLM_FALLBACK = auto()

    # --- tool branch ---
    TOOL_ACTION = auto()
    EXECUTING = auto()
    CHAINING = auto()
    TOOL_DONE = auto()

    # --- workflow branch ---
    WORKFLOW_ACTIVE = auto()
    STEP_EXEC = auto()
    WAIT_USER = auto()
    WF_DONE = auto()
    SUSPENDED = auto()

    # --- terminal ---
    HUMAN_TRANSFER = auto()
    FAREWELL = auto()


# ------------------------------------------------------------------
# transition record kept for debugging
# ------------------------------------------------------------------

class _Transition(NamedTuple):
    from_state: DialogueState
    to_state: DialogueState
    trigger: str
    timestamp: float


# ------------------------------------------------------------------
# transition table
# ------------------------------------------------------------------

_TRANSITIONS: dict[tuple[DialogueState, str], DialogueState] = {
    # --- opening / auth ---
    (DialogueState.GREETING, "ask_for_id"):              DialogueState.AUTHENTICATING,

    (DialogueState.AUTHENTICATING, "verify_attempt"):     DialogueState.VERIFY,

    (DialogueState.VERIFY, "pass"):                      DialogueState.VERIFIED,
    (DialogueState.VERIFY, "need_more"):                 DialogueState.COLLECT_ID,
    (DialogueState.VERIFY, "fail_3x"):                   DialogueState.LOCKED,

    (DialogueState.COLLECT_ID, "verify_attempt"):        DialogueState.VERIFY,

    (DialogueState.LOCKED, "locked"):                    DialogueState.HUMAN_TRANSFER,

    # --- auth -> conversation ---
    (DialogueState.VERIFIED, "start_listening"):         DialogueState.LISTENING,
    (DialogueState.LISTENING, "ready"):                  DialogueState.READY,

    # --- READY hub: intent routing ---
    (DialogueState.READY, "policy_question"):            DialogueState.FAQ_SEARCHING,
    (DialogueState.READY, "needs_data"):                 DialogueState.EXECUTING,
    (DialogueState.READY, "multi_step"):                 DialogueState.STEP_EXEC,
    (DialogueState.READY, "ambiguous"):                  DialogueState.CLARIFYING,
    (DialogueState.READY, "goodbye"):                    DialogueState.FAREWELL,
    (DialogueState.READY, "demands_agent"):              DialogueState.HUMAN_TRANSFER,
    (DialogueState.READY, "chit_chat"):                  DialogueState.READY,

    # --- FAQ branch ---
    (DialogueState.FAQ_SEARCHING, "high_confidence"):    DialogueState.PURE_FAQ,
    (DialogueState.FAQ_SEARCHING, "medium_confidence"):  DialogueState.BLENDED_FAQ,
    (DialogueState.FAQ_SEARCHING, "low_confidence"):     DialogueState.LLM_FALLBACK,

    (DialogueState.PURE_FAQ, "answered"):                DialogueState.READY,
    (DialogueState.BLENDED_FAQ, "answered"):             DialogueState.READY,
    (DialogueState.LLM_FALLBACK, "answered"):            DialogueState.READY,

    # --- tool branch ---
    (DialogueState.EXECUTING, "complete"):               DialogueState.TOOL_DONE,
    (DialogueState.EXECUTING, "needs_follow_up"):        DialogueState.CHAINING,

    (DialogueState.CHAINING, "next_tool"):               DialogueState.EXECUTING,

    (DialogueState.TOOL_DONE, "answered"):               DialogueState.READY,

    # --- workflow branch ---
    (DialogueState.STEP_EXEC, "complete"):               DialogueState.WF_DONE,
    (DialogueState.STEP_EXEC, "needs_input"):            DialogueState.WAIT_USER,
    (DialogueState.STEP_EXEC, "auto_advance"):           DialogueState.STEP_EXEC,

    (DialogueState.WAIT_USER, "user_responds"):          DialogueState.STEP_EXEC,
    (DialogueState.WAIT_USER, "user_switches_topic"):    DialogueState.SUSPENDED,

    (DialogueState.WF_DONE, "complete"):                 DialogueState.READY,

    # --- suspended workflow ---
    (DialogueState.SUSPENDED, "user_returns"):           DialogueState.STEP_EXEC,
    (DialogueState.SUSPENDED, "expired"):                DialogueState.READY,

    # --- side-questions from workflow/tool contexts ---
    (DialogueState.SUSPENDED, "side_question_faq"):      DialogueState.FAQ_SEARCHING,
    (DialogueState.SUSPENDED, "side_question_data"):     DialogueState.EXECUTING,

    # --- clarification ---
    (DialogueState.CLARIFYING, "clarified"):             DialogueState.READY,
}


class InvalidTransitionError(Exception):
    """Raised when a trigger is not valid for the current state."""


# ------------------------------------------------------------------
# state machine
# ------------------------------------------------------------------

class DialogueStateMachine:
    """Deterministic finite-state machine for dialogue management.

    The machine starts at ``GREETING`` and moves through the
    conversation lifecycle via explicit ``transition(trigger)`` calls.
    Every (state, trigger) pair must exist in the transition table;
    unrecognised triggers raise ``InvalidTransitionError``.
    """

    def __init__(self) -> None:
        self._state: DialogueState = DialogueState.GREETING
        self._history: list[_Transition] = []

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> DialogueState:
        """The state the machine is currently in."""
        return self._state

    @property
    def history(self) -> list[_Transition]:
        """Full transition log for debugging."""
        return list(self._history)

    def can_transition(self, trigger: str) -> bool:
        """Return ``True`` if *trigger* is valid from the current state."""
        return (self._state, trigger) in _TRANSITIONS

    def transition(self, trigger: str) -> DialogueState:
        """Apply *trigger* and return the new state.

        Raises ``InvalidTransitionError`` if the trigger is not valid
        for the current state.
        """
        key = (self._state, trigger)
        if key not in _TRANSITIONS:
            raise InvalidTransitionError(
                f"no transition for trigger {trigger!r} "
                f"in state {self._state.name}"
            )

        prev = self._state
        self._state = _TRANSITIONS[key]
        self._history.append(
            _Transition(
                from_state=prev,
                to_state=self._state,
                trigger=trigger,
                timestamp=time.time(),
            )
        )
        return self._state

    def valid_triggers(self) -> list[str]:
        """Return every trigger that is valid from the current state."""
        return [
            trigger
            for (state, trigger) in _TRANSITIONS
            if state == self._state
        ]

    # ------------------------------------------------------------------
    # convenience queries
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """True when the machine is in a terminal state."""
        return self._state in (
            DialogueState.FAREWELL,
            DialogueState.HUMAN_TRANSFER,
        )

    @property
    def is_authenticated(self) -> bool:
        """True after verification has passed."""
        return self._state not in (
            DialogueState.GREETING,
            DialogueState.AUTHENTICATING,
            DialogueState.COLLECT_ID,
            DialogueState.VERIFY,
            DialogueState.LOCKED,
        )

    def __repr__(self) -> str:
        return f"DialogueStateMachine(state={self._state.name})"
