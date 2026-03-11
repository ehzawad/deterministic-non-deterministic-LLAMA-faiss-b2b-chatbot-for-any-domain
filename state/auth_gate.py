"""Authentication gate -- guards every session behind phone verification.

Coordinates between the tool executor (which calls the verify_phone
tool), the session store (which gets sealed on success), and the
dialogue state machine (which tracks auth progress).
"""

from __future__ import annotations

from typing import Any, Protocol

from memory.fact_store import FactStore
from memory.session_store import SessionStore
from state.dialogue_state_machine import DialogueState, DialogueStateMachine


# ------------------------------------------------------------------
# protocol for the tool executor dependency
# ------------------------------------------------------------------

class ToolExecutorProtocol(Protocol):
    """Minimal interface the auth gate needs from a tool executor."""

    def execute(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]: ...


# ------------------------------------------------------------------
# auth gate
# ------------------------------------------------------------------

class AuthGate:
    """Guards entry to the main conversation behind phone verification.

    Typical flow::

        gate = AuthGate(tool_executor, session_store, state_machine)
        greeting = gate.start_auth()          # -> AUTHENTICATING -> VERIFY
        ok, msg = gate.attempt_verify(phone, fact_store, turn)
        # if ok  -> session_store is sealed, state is VERIFIED
        # if not -> attempts incremented; after 3 -> LOCKED -> HUMAN_TRANSFER
    """

    def __init__(
        self,
        tool_executor: ToolExecutorProtocol,
        session_store: SessionStore,
        state_machine: DialogueStateMachine,
        *,
        max_attempts: int = 3,
    ) -> None:
        self._tool_executor = tool_executor
        self._session_store = session_store
        self._sm = state_machine
        self._max_attempts = max_attempts
        self._attempts = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def start_auth(self) -> str:
        """Issue the greeting and ask for a phone number.

        Transitions the state machine: GREETING -> AUTHENTICATING -> VERIFY.
        Returns a greeting message the caller can send to the user.
        """
        self._sm.transition("ask_for_id")
        self._sm.transition("verify_attempt")
        return (
            "Welcome! I can help you with your account today. "
            "To get started, could you please provide the phone number "
            "associated with your account?"
        )

    def attempt_verify(
        self,
        phone: str,
        fact_store: FactStore,
        turn_number: int,
    ) -> tuple[bool, str]:
        """Try to verify *phone* via the tool executor.

        Returns ``(success, message)``.  On success the session store is
        sealed and the machine moves to VERIFIED.  On failure the attempt
        counter increments; after *max_attempts* failures the machine
        transitions to LOCKED then HUMAN_TRANSFER.
        """
        # If already locked, refuse immediately.
        if self.is_locked:
            return False, (
                "Your account has been locked after too many failed attempts. "
                "I am transferring you to a human agent now."
            )

        # Call the verify_phone tool via the batch execute interface.
        results = self._tool_executor.execute(
            [{"function": "verify_phone", "params": {"phone": phone}}],
            fact_store=fact_store,
            turn_number=turn_number,
        )

        result = results[0].get("result", {}) if results else {}
        verified: bool = result.get("verified", False)

        if verified:
            return self._handle_success(result, phone, fact_store, turn_number)
        return self._handle_failure(fact_store, turn_number)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """True when the account has been locked due to failed attempts."""
        return self._sm.current_state in (
            DialogueState.LOCKED,
            DialogueState.HUMAN_TRANSFER,
        )

    @property
    def is_verified(self) -> bool:
        """True after successful verification."""
        return self._session_store.is_sealed and self._sm.current_state not in (
            DialogueState.GREETING,
            DialogueState.AUTHENTICATING,
            DialogueState.COLLECT_ID,
            DialogueState.VERIFY,
            DialogueState.LOCKED,
        )

    @property
    def attempts_remaining(self) -> int:
        return max(0, self._max_attempts - self._attempts)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _handle_success(
        self,
        result: dict[str, Any],
        phone: str,
        fact_store: FactStore,
        turn_number: int,
    ) -> tuple[bool, str]:
        """Seal the session, record facts, advance the state machine."""
        name: str = result.get("name", "Customer")
        account_type: str = result.get("account_type", "standard")
        tenure: str = result.get("tenure", "unknown")

        # Seal session store -- one-time write.
        self._session_store.seal(
            name=name,
            account_type=account_type,
            verified=True,
            tenure=tenure,
            phone=phone,
        )

        # Record auth facts for the context window.
        for key, value in (
            ("customer_name", name),
            ("account_type", account_type),
            ("tenure", tenure),
            ("phone", phone),
        ):
            fact_store.add(
                key=key,
                value=value,
                turn_number=turn_number,
                source_tool="verify_phone",
            )

        # State: VERIFY -> VERIFIED.
        self._sm.transition("pass")

        return True, f"Great, I have verified your identity. Welcome, {name}!"

    def _handle_failure(
        self,
        fact_store: FactStore,
        turn_number: int,
    ) -> tuple[bool, str]:
        """Increment the attempt counter; lock if exhausted."""
        self._attempts += 1

        fact_store.add(
            key="auth_attempt",
            value=f"failed (attempt {self._attempts}/{self._max_attempts})",
            turn_number=turn_number,
            source_tool="verify_phone",
        )

        if self._attempts >= self._max_attempts:
            # State: VERIFY -> LOCKED -> HUMAN_TRANSFER.
            self._sm.transition("fail_3x")
            self._sm.transition("locked")
            return False, (
                "I am sorry, but I was unable to verify your identity after "
                f"{self._max_attempts} attempts. For your security I am "
                "transferring you to a human agent."
            )

        # State: VERIFY -> COLLECT_ID (ask for more info).
        self._sm.transition("need_more")

        remaining = self._max_attempts - self._attempts
        return False, (
            "I was not able to verify that phone number. "
            f"You have {remaining} attempt{'s' if remaining != 1 else ''} remaining. "
            "Could you please try again?"
        )
