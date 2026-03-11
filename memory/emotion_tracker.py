"""Store 4 -- Emotion trajectory tracker.

Records one emotion label per turn and provides escalation detection
so the system can route to a human when frustration is climbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import EmotionConfig

_CFG = EmotionConfig()

# Build an index so we can compare emotion intensity numerically.
_EMOTION_INDEX: dict[str, int] = {e: i for i, e in enumerate(_CFG.LEVELS)}


@dataclass(frozen=True)
class EmotionSample:
    """A single emotion reading at a given turn."""

    turn_number: int
    emotion: str


class EmotionTracker:
    """Tracks the emotional trajectory of a conversation."""

    def __init__(self) -> None:
        self._trajectory: list[EmotionSample] = []

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    def record(self, turn_number: int, emotion: str) -> None:
        """Append an emotion reading.

        Raises ``ValueError`` if *emotion* is not one of the known
        levels defined in ``EmotionConfig.LEVELS``.
        """
        if emotion not in _EMOTION_INDEX:
            raise ValueError(
                f"Unknown emotion '{emotion}'; expected one of {_CFG.LEVELS}"
            )
        self._trajectory.append(EmotionSample(turn_number=turn_number, emotion=emotion))

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    @property
    def latest(self) -> EmotionSample | None:
        return self._trajectory[-1] if self._trajectory else None

    @property
    def trajectory(self) -> list[EmotionSample]:
        return list(self._trajectory)

    def is_escalating(self, window: int | None = None) -> bool:
        """Return ``True`` if the last *window* emotions are strictly increasing.

        Uses ``EmotionConfig.ESCALATION_WINDOW`` when *window* is not
        provided.  Returns ``False`` if there are fewer samples than
        the window size.
        """
        if window is None:
            window = _CFG.ESCALATION_WINDOW

        if len(self._trajectory) < window:
            return False

        recent = self._trajectory[-window:]
        levels = [_EMOTION_INDEX[s.emotion] for s in recent]

        return all(levels[i] < levels[i + 1] for i in range(len(levels) - 1))

    def __len__(self) -> int:
        return len(self._trajectory)

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    def to_context_string(self) -> str:
        """Render for injection into the LLM context window."""
        if not self._trajectory:
            return "[emotion: no readings]"

        arrow = " -> ".join(s.emotion for s in self._trajectory)
        escalating_flag = " !!ESCALATING" if self.is_escalating() else ""
        return f"[emotion: {arrow}{escalating_flag}]"
