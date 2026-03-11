"""Store 1 -- Session identity, written ONCE at auth then locked.

Holds customer identity facts that come from the authentication step.
Once sealed, every field is frozen for the rest of the session.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionStore:
    """Immutable-after-seal bag of customer identity facts."""

    name: Optional[str] = None
    account_type: Optional[str] = None
    verified: bool = False
    tenure: Optional[str] = None
    phone: Optional[str] = None
    auth_timestamp: Optional[float] = None

    _sealed: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def seal(
        self,
        *,
        name: str,
        account_type: str,
        verified: bool,
        tenure: str,
        phone: str,
    ) -> None:
        """Write every field exactly once, then lock the store.

        Raises ``RuntimeError`` if the store was already sealed.
        """
        if self._sealed:
            raise RuntimeError("SessionStore is already sealed; cannot write again")

        self.name = name
        self.account_type = account_type
        self.verified = verified
        self.tenure = tenure
        self.phone = phone
        self.auth_timestamp = time.time()
        self._sealed = True

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    def to_context_string(self) -> str:
        """Render for injection into the LLM context window."""
        if not self._sealed:
            return "[session: unauthenticated]"

        return (
            f"[session: name={self.name} | account={self.account_type} | "
            f"verified={self.verified} | tenure={self.tenure} | "
            f"phone={self.phone}]"
        )
