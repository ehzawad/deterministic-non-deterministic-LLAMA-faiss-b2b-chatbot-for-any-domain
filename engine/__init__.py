"""engine -- NLP engine components (agent model, generator model).

Usage::

    from engine import (
        DualLLMEngine,
        ContextBuilder,
        Pass1Resolver, Pass1Result,
        Pass2Naturalizer,
        ActionRouter, TurnResult,
    )
"""

from engine.llm import DualLLMEngine
from engine.context_builder import ContextBuilder
from engine.pass1_resolver import Pass1Resolver, Pass1Result
from engine.pass2_naturalizer import Pass2Naturalizer
from engine.action_router import ActionRouter, TurnResult

__all__ = [
    "DualLLMEngine",
    "ContextBuilder",
    "Pass1Resolver",
    "Pass1Result",
    "Pass2Naturalizer",
    "ActionRouter",
    "TurnResult",
]
