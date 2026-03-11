"""Store 5 -- Ring-buffered turn history.

Keeps the last N turns (default 16, configurable via ``MemoryConfig``)
in a ``collections.deque`` so old turns fall off automatically.
Every turn is stored verbatim -- no summarisation, no truncation.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from config import MemoryConfig

_CFG = MemoryConfig()


@dataclass
class Turn:
    """One conversational turn."""

    turn_number: int
    role: str  # "user" | "assistant" | "tool"
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


class TurnHistory:
    """Fixed-size ring buffer of raw conversation turns.

    Oldest turns are silently evicted when the buffer is full.
    """

    def __init__(self, maxlen: int | None = None) -> None:
        size = maxlen if maxlen is not None else _CFG.MAX_TURN_HISTORY
        self._buffer: collections.deque[Turn] = collections.deque(maxlen=size)
        self._turn_counter: int = 0

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    def add(
        self,
        *,
        turn_number: int,
        role: str,
        text: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> Turn:
        """Append a turn to the ring buffer."""
        turn = Turn(
            turn_number=turn_number,
            role=role,
            text=text,
            tool_calls=tool_calls if tool_calls is not None else [],
            tool_results=tool_results if tool_results is not None else [],
        )
        self._buffer.append(turn)
        self._turn_counter += 1
        return turn

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    @property
    def turns(self) -> list[Turn]:
        return list(self._buffer)

    @property
    def last(self) -> Turn | None:
        return self._buffer[-1] if self._buffer else None

    @property
    def turn_counter(self) -> int:
        """Cumulative count of all turns ever added (not just buffer size)."""
        return self._turn_counter

    @property
    def maxlen(self) -> int:
        return self._buffer.maxlen  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    def _render_turn(self, turn: Turn) -> str:
        parts = [f"[T{turn.turn_number} {turn.role}] {turn.text}"]
        if turn.tool_calls:
            parts.append(f"  tool_calls: {turn.tool_calls}")
        if turn.tool_results:
            parts.append(f"  tool_results: {turn.tool_results}")
        return "\n".join(parts)

    def to_context_string(self) -> str:
        """Return raw verbatim turns for injection into the context window."""
        if not self._buffer:
            return "[history: empty]"

        rendered = "\n".join(self._render_turn(t) for t in self._buffer)
        return f"[history]\n{rendered}\n[/history]"
