"""G3 -- Fact-integrity gate (architectural enforcement).

G3 is enforced primarily by code structure: only ``ToolExecutor`` writes
to the fact store.  This module provides a small marker / verification
helper that other components can call when they want to assert the
invariant explicitly.
"""

from __future__ import annotations

# The sole authorised caller that is allowed to write facts.
_AUTHORISED_CALLER = "tool_executor"


def verify_write_source(caller: str) -> bool:
    """Return ``True`` only when *caller* is the authorised fact writer.

    In practice the fact store is written inside
    ``ToolExecutor.execute()``, so this function is a documentation-level
    guard that can be invoked from tests or auditing hooks.

    >>> verify_write_source("tool_executor")
    True
    >>> verify_write_source("llm_agent")
    False
    """
    return caller == _AUTHORISED_CALLER
