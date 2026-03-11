"""G4 -- Hedge enforcer for medium-confidence FAQ responses.

When the FAQ engine returns a *hedged* response (medium confidence), the
LLM must not sound overly certain.  This guardrail strips forbidden
absolute words and, when necessary, replaces them with softer hedges.
"""

from __future__ import annotations

import re

# ── vocabulary ────────────────────────────────────────────────────

FORBIDDEN: list[str] = [
    "definitely",
    "always",
    "guaranteed",
    "certainly",
    "absolutely",
    "for sure",
    "100%",
    "without a doubt",
]

REQUIRED_HEDGES: list[str] = [
    "typically",
    "usually",
    "in most cases",
    "generally",
    "normally",
    "as a general rule",
]

# Map each forbidden word to a preferred hedge replacement.
_REPLACEMENTS: dict[str, str] = {
    "definitely":       "typically",
    "always":           "usually",
    "guaranteed":       "generally",
    "certainly":        "normally",
    "absolutely":       "generally",
    "for sure":         "in most cases",
    "100%":             "in most cases",
    "without a doubt":  "as a general rule",
}

# Pre-compile a single regex that matches any forbidden term
# (case-insensitive, whole-word where possible).
_FORBIDDEN_RE = re.compile(
    "|".join(re.escape(word) for word in FORBIDDEN),
    re.IGNORECASE,
)


def check(response: str, hedged: bool) -> tuple[bool, list[str]]:
    """Check *response* for hedge-policy violations.

    Only active when *hedged* is ``True`` (medium-confidence FAQ).

    Returns:
        ``(True, [])`` when the response is clean, or
        ``(False, [list of violations])`` when forbidden words are found.
    """
    if not hedged:
        return True, []

    violations: list[str] = []
    for match in _FORBIDDEN_RE.finditer(response):
        violations.append(match.group())

    passed = len(violations) == 0
    return passed, violations


def fix(response: str) -> str:
    """Replace every forbidden word in *response* with its hedge alternative."""

    def _replace(match: re.Match) -> str:
        word = match.group()
        # Look up case-insensitively.
        return _REPLACEMENTS.get(word.lower(), "generally")

    return _FORBIDDEN_RE.sub(_replace, response)
