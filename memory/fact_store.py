"""Store 2 -- Versioned fact store, written ONLY by tool results.

Every fact is append-only: updates create a new version so we never
lose history.  Each entry records the tool that produced it and the
turn at which it arrived.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fact:
    """A single versioned fact produced by a tool."""

    key: str
    value: str
    turn_number: int
    timestamp: float
    source_tool: str


class FactStore:
    """Append-only, versioned collection of tool-produced facts.

    Internally facts are stored in a ``dict[str, list[Fact]]`` so that
    looking up the *latest* value for a key is O(1) while the full
    version chain is preserved.
    """

    def __init__(self) -> None:
        self._facts: dict[str, list[Fact]] = {}

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    def add(
        self,
        *,
        key: str,
        value: str,
        turn_number: int,
        source_tool: str,
    ) -> Fact:
        """Record a new fact (or a new version of an existing key)."""
        fact = Fact(
            key=key,
            value=value,
            turn_number=turn_number,
            timestamp=time.time(),
            source_tool=source_tool,
        )
        self._facts.setdefault(key, []).append(fact)
        return fact

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    def latest(self, key: str) -> Fact | None:
        """Return the most recent version of *key*, or ``None``."""
        chain = self._facts.get(key)
        return chain[-1] if chain else None

    def all_versions(self, key: str) -> list[Fact]:
        """Return the full version chain for *key*."""
        return list(self._facts.get(key, []))

    def all_latest(self) -> list[Fact]:
        """Return the latest version of every key."""
        return [versions[-1] for versions in self._facts.values()]

    @property
    def keys(self) -> list[str]:
        return list(self._facts.keys())

    def __len__(self) -> int:
        return len(self._facts)

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    def to_context_string(self) -> str:
        """Render for injection into the LLM context window."""
        if not self._facts:
            return "[facts: none]"

        parts: list[str] = []
        for fact in self.all_latest():
            parts.append(f"{fact.key}={fact.value} @turn{fact.turn_number}")
        return "[facts: " + " | ".join(parts) + "]"
