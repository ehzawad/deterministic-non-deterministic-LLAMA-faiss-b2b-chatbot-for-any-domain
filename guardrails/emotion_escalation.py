"""G5 -- Emotion-escalation detector.

Monitors the emotion tracker for a rising frustration trend.  When the
last N readings form a strictly increasing sequence (using the numeric
emotion index), we offer to route the customer to a human agent.
"""

from __future__ import annotations

from memory.emotion_tracker import EmotionTracker

ESCALATION_MESSAGE = (
    "I can sense this is frustrating. Would you like me to connect you "
    "with a human agent who might be able to help more directly?"
)


def check(emotion_tracker: EmotionTracker) -> tuple[bool, str]:
    """Return ``(escalating, message)``.

    *escalating* is ``True`` when the emotion trajectory shows a
    sustained rise over the configured window.  *message* contains the
    human-handoff offer when escalating, or an empty string otherwise.
    """
    if emotion_tracker.is_escalating():
        return True, ESCALATION_MESSAGE

    return False, ""
