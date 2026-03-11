"""Store 3 -- Thread tracker for multi-topic conversations.

Each thread is a named topic with:
  - a status  (ACTIVE / SUSPENDED / RESOLVED)
  - slots_filled  -- information already gathered
  - slots_needed  -- information still required

Only one thread should be ACTIVE at a time.  Others are SUSPENDED
until the user returns to them, or RESOLVED once handled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ThreadStatus(Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    RESOLVED = "RESOLVED"


@dataclass
class Thread:
    """A single conversation topic being tracked."""

    topic: str
    status: ThreadStatus = ThreadStatus.ACTIVE
    slots_filled: dict[str, Any] = field(default_factory=dict)
    slots_needed: list[str] = field(default_factory=list)


class ThreadTracker:
    """Manages multiple conversation threads/topics."""

    def __init__(self) -> None:
        self._threads: dict[str, Thread] = {}

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def open(self, topic: str, slots_needed: list[str] | None = None) -> Thread:
        """Open a new thread (or reactivate an existing one).

        If a thread with this topic already exists and is SUSPENDED,
        it is resumed instead of creating a duplicate.
        """
        existing = self._threads.get(topic)
        if existing is not None:
            existing.status = ThreadStatus.ACTIVE
            if slots_needed is not None:
                # merge in any newly-discovered slots
                for s in slots_needed:
                    if s not in existing.slots_needed and s not in existing.slots_filled:
                        existing.slots_needed.append(s)
            return existing

        thread = Thread(
            topic=topic,
            slots_needed=list(slots_needed) if slots_needed else [],
        )
        self._threads[topic] = thread
        return thread

    def fill_slot(self, topic: str, slot: str, value: Any) -> None:
        """Record that *slot* has been filled with *value*."""
        thread = self._threads.get(topic)
        if thread is None:
            raise KeyError(f"No thread with topic '{topic}'")
        thread.slots_filled[slot] = value
        if slot in thread.slots_needed:
            thread.slots_needed.remove(slot)

    def suspend(self, topic: str) -> None:
        """Park a thread -- the user switched topics."""
        thread = self._threads.get(topic)
        if thread is None:
            raise KeyError(f"No thread with topic '{topic}'")
        thread.status = ThreadStatus.SUSPENDED

    def resolve(self, topic: str) -> None:
        """Mark a thread as done."""
        thread = self._threads.get(topic)
        if thread is None:
            raise KeyError(f"No thread with topic '{topic}'")
        thread.status = ThreadStatus.RESOLVED

    def resume(self, topic: str) -> Thread:
        """Bring a SUSPENDED thread back to ACTIVE."""
        thread = self._threads.get(topic)
        if thread is None:
            raise KeyError(f"No thread with topic '{topic}'")
        thread.status = ThreadStatus.ACTIVE
        return thread

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    @property
    def active(self) -> Thread | None:
        """Return the currently active thread, if any."""
        for t in self._threads.values():
            if t.status == ThreadStatus.ACTIVE:
                return t
        return None

    @property
    def suspended(self) -> list[Thread]:
        return [t for t in self._threads.values() if t.status == ThreadStatus.SUSPENDED]

    @property
    def resolved(self) -> list[Thread]:
        return [t for t in self._threads.values() if t.status == ThreadStatus.RESOLVED]

    @property
    def all_threads(self) -> list[Thread]:
        return list(self._threads.values())

    def __len__(self) -> int:
        return len(self._threads)

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    def _render_thread(self, thread: Thread) -> str:
        filled = ", ".join(f"{k}={v}" for k, v in thread.slots_filled.items()) or "none"
        needed = ", ".join(thread.slots_needed) or "none"
        return (
            f"{thread.topic}({thread.status.value}) "
            f"filled=[{filled}] needed=[{needed}]"
        )

    def to_context_string(self) -> str:
        """Render for injection into the LLM context window."""
        if not self._threads:
            return "[threads: none]"

        parts = [self._render_thread(t) for t in self._threads.values()]
        return "[threads: " + " | ".join(parts) + "]"
