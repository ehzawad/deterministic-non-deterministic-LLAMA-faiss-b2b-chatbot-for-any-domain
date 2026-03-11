#!/usr/bin/env python3
"""End-to-end test replaying the D10 banking scenario.

Simulates the full conversation pipeline without interactive input.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    AgentModelConfig,
    GeneratorModelConfig,
    EmbeddingConfig,
    FAQConfig,
    MemoryConfig,
    AuthConfig,
    EmotionConfig,
    WorkflowConfig,
)


def main() -> None:
    print("=" * 60)
    print("  D10 Scenario Replay — Full Pipeline Test")
    print("=" * 60)

    # ── Load everything ───────────────────────────────────────────
    print("\n[Loading models...]")
    from engine.llm import DualLLMEngine
    engine = DualLLMEngine(AgentModelConfig(), GeneratorModelConfig())
    print("  LLMs loaded.")

    print("[Loading FAQ...]")
    from faq.embedder import Embedder
    from faq.faq_engine import FAQEngine
    embedder = Embedder(EmbeddingConfig().MODEL_NAME, EmbeddingConfig().DEVICE)
    faq_engine = FAQEngine(embedder, FAQConfig())
    print(f"  {faq_engine.index.ntotal} FAQ entries.")

    print("[Initializing components...]")
    from memory.session_store import SessionStore
    from memory.fact_store import FactStore
    from memory.thread_tracker import ThreadTracker
    from memory.emotion_tracker import EmotionTracker
    from memory.turn_history import TurnHistory
    from tools.tool_registry import TOOL_REGISTRY
    from tools import mock_tools
    from tools.tool_executor import ToolExecutor
    from workflow.workflow_engine import WorkflowEngine
    from workflow.workflow_definitions import WORKFLOW_REGISTRY
    from state.dialogue_state_machine import DialogueStateMachine
    from state.auth_gate import AuthGate
    from guardrails.guardrails import GuardrailsPipeline
    from engine.context_builder import ContextBuilder
    from engine.pass1_resolver import Pass1Resolver
    from engine.pass2_naturalizer import Pass2Naturalizer
    from engine.action_router import ActionRouter

    session_store = SessionStore()
    fact_store = FactStore()
    thread_tracker = ThreadTracker()
    emotion_tracker = EmotionTracker()
    turn_history = TurnHistory(maxlen=MemoryConfig().MAX_TURN_HISTORY)

    tool_executor = ToolExecutor(TOOL_REGISTRY, mock_tools)
    workflow_engine = WorkflowEngine()
    state_machine = DialogueStateMachine()
    auth_gate = AuthGate(tool_executor, session_store, state_machine, max_attempts=AuthConfig().MAX_AUTH_ATTEMPTS)
    guardrails = GuardrailsPipeline(fact_store, emotion_tracker)

    tool_defs = [
        {"name": n, "description": t.description, "parameters": t.params_schema}
        for n, t in TOOL_REGISTRY.items()
    ]
    context_builder = ContextBuilder(engine, tool_defs)
    pass1 = Pass1Resolver(engine, context_builder)
    pass2 = Pass2Naturalizer(engine, context_builder)

    action_router = ActionRouter(
        pass1=pass1, pass2=pass2, faq_engine=faq_engine,
        tool_executor=tool_executor, workflow_engine=workflow_engine,
        guardrails=guardrails, state_machine=state_machine,
        session_store=session_store, fact_store=fact_store,
        thread_tracker=thread_tracker, emotion_tracker=emotion_tracker,
        turn_history=turn_history,
    )
    print("  All components ready.\n")

    # ── Authentication ────────────────────────────────────────────
    print("-" * 60)
    greeting = auth_gate.start_auth()
    print(f"Bot: {greeting}")

    # Simulate auth with phone number
    phone = "01712345678"
    print(f"User: {phone}")
    turn_num = len(turn_history.turns) + 1
    turn_history.add(turn_number=turn_num, role="user", text=phone)
    success, msg = auth_gate.attempt_verify(phone, fact_store, turn_num)
    print(f"Bot: {msg}")
    turn_history.add(turn_number=turn_num + 1, role="assistant", text=msg)

    if not success:
        print("AUTH FAILED - aborting test")
        return

    state_machine.transition("start_listening")
    state_machine.transition("ready")
    print(f"[Auth OK, state: {state_machine.current_state}]")

    # ── D10 conversation turns ────────────────────────────────────
    test_turns = [
        "Am I eligible for a credit card?",
        "what documents do I need?",
        "I dont have salary slip",
        "when will I get the card?",
        "actually whats my balance?",
        "is that enough to get approved?",
    ]

    print("-" * 60)
    for utterance in test_turns:
        print(f"\nUser: {utterance}")
        result = action_router.process_turn(utterance)

        if result.filler_message:
            print(f"Bot (filler): {result.filler_message}")
        print(f"Bot: {result.final_response}")
        print(f"  [action={result.action_type}, state={state_machine.current_state}, emotion={emotion_tracker.latest}]")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    print(f"  Total turns: {len(turn_history.turns)}")
    print(f"  State: {state_machine.current_state}")
    print(f"  Emotion: {emotion_tracker.to_context_string()}")
    print(f"  Facts: {fact_store.to_context_string()}")
    print(f"  Session: {session_store.to_context_string()}")
    print()
    print("  TEST COMPLETE")


if __name__ == "__main__":
    main()
