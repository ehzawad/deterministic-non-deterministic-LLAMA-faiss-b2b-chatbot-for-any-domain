#!/usr/bin/env python3
"""Five realistic call center / web chat conversation scenarios.

Each scenario simulates a real customer interaction that would happen
in a Bangladeshi bank's support center — either via voice call or web chat.
The NLP pipeline processes each turn and we print the full exchange.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_system():
    """Initialize the full chatbot system once."""
    from config import (
        AgentModelConfig, GeneratorModelConfig, EmbeddingConfig,
        FAQConfig, MemoryConfig, AuthConfig,
    )
    from engine.llm import DualLLMEngine
    from faq.embedder import Embedder
    from faq.faq_engine import FAQEngine
    from memory.session_store import SessionStore
    from memory.fact_store import FactStore
    from memory.thread_tracker import ThreadTracker
    from memory.emotion_tracker import EmotionTracker
    from memory.turn_history import TurnHistory
    from tools.tool_registry import TOOL_REGISTRY
    from tools import mock_tools
    from tools.tool_executor import ToolExecutor
    from workflow.workflow_engine import WorkflowEngine
    from state.dialogue_state_machine import DialogueStateMachine
    from state.auth_gate import AuthGate
    from guardrails.guardrails import GuardrailsPipeline
    from engine.context_builder import ContextBuilder
    from engine.pass1_resolver import Pass1Resolver
    from engine.pass2_naturalizer import Pass2Naturalizer
    from engine.action_router import ActionRouter

    print("Loading models...")
    engine = DualLLMEngine(AgentModelConfig(), GeneratorModelConfig())
    embedder = Embedder(EmbeddingConfig().MODEL_NAME, EmbeddingConfig().DEVICE)
    faq_engine = FAQEngine(embedder, FAQConfig())
    print(f"Models loaded. {faq_engine.index.ntotal} FAQ entries.\n")

    tool_executor = ToolExecutor(TOOL_REGISTRY, mock_tools)
    tool_defs = [
        {"name": n, "description": t.description, "parameters": t.params_schema}
        for n, t in TOOL_REGISTRY.items()
    ]
    context_builder = ContextBuilder(engine, tool_defs)
    pass1 = Pass1Resolver(engine, context_builder)
    pass2 = Pass2Naturalizer(engine, context_builder)

    return {
        "engine": engine, "faq_engine": faq_engine,
        "tool_executor": tool_executor, "context_builder": context_builder,
        "pass1": pass1, "pass2": pass2, "TOOL_REGISTRY": TOOL_REGISTRY,
        "mock_tools": mock_tools,
    }


def run_scenario(system, scenario_name, phone, turns):
    """Run a single conversation scenario end-to-end."""
    from memory.session_store import SessionStore
    from memory.fact_store import FactStore
    from memory.thread_tracker import ThreadTracker
    from memory.emotion_tracker import EmotionTracker
    from memory.turn_history import TurnHistory
    from state.dialogue_state_machine import DialogueStateMachine
    from state.auth_gate import AuthGate
    from guardrails.guardrails import GuardrailsPipeline
    from engine.action_router import ActionRouter
    from config import MemoryConfig, AuthConfig
    from workflow.workflow_engine import WorkflowEngine

    # Fresh state per scenario
    session_store = SessionStore()
    fact_store = FactStore()
    thread_tracker = ThreadTracker()
    emotion_tracker = EmotionTracker()
    turn_history = TurnHistory(maxlen=MemoryConfig().MAX_TURN_HISTORY)
    state_machine = DialogueStateMachine()
    auth_gate = AuthGate(
        system["tool_executor"], session_store, state_machine,
        max_attempts=AuthConfig().MAX_AUTH_ATTEMPTS,
    )
    guardrails = GuardrailsPipeline(fact_store, emotion_tracker)
    workflow_engine = WorkflowEngine()

    action_router = ActionRouter(
        pass1=system["pass1"], pass2=system["pass2"],
        faq_engine=system["faq_engine"],
        tool_executor=system["tool_executor"],
        workflow_engine=workflow_engine,
        guardrails=guardrails, state_machine=state_machine,
        session_store=session_store, fact_store=fact_store,
        thread_tracker=thread_tracker, emotion_tracker=emotion_tracker,
        turn_history=turn_history,
    )

    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'='*70}")

    # Auth phase
    greeting = auth_gate.start_auth()
    print(f"\nBot: {greeting}")
    print(f"Customer: {phone}")

    turn_num = 1
    turn_history.add(turn_number=turn_num, role="user", text=phone)
    success, msg = auth_gate.attempt_verify(phone, fact_store, turn_num)
    print(f"Bot: {msg}")
    turn_history.add(turn_number=turn_num + 1, role="assistant", text=msg)

    if not success:
        print("[AUTH FAILED]")
        return

    state_machine.transition("start_listening")
    state_machine.transition("ready")
    print(f"  [Authenticated as: {session_store.name}]\n")

    # Conversation turns
    for i, utterance in enumerate(turns, 1):
        print(f"Customer: {utterance}")
        t0 = time.time()
        result = action_router.process_turn(utterance)
        elapsed = time.time() - t0

        if result.filler_message and result.filler_message != "string":
            print(f"Bot: {result.filler_message}")
        print(f"Bot: {result.final_response}")
        print(f"  [{result.action_type} | {elapsed:.1f}s]")
        print()

    # Summary
    print(f"  --- Scenario Summary ---")
    print(f"  Facts: {fact_store.to_context_string()}")
    print(f"  Emotion: {emotion_tracker.to_context_string()}")


# ═══════════════════════════════════════════════════════════════════
# FIVE REALISTIC SCENARIOS
# ═══════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "name": "Scenario 1: New Customer Inquiring About Credit Card (Web Chat)",
        "phone": "01712345678",
        "turns": [
            "I want to apply for a credit card",
            "What are the requirements?",
            "What documents will I need to submit?",
            "How long will it take to get the card after approval?",
            "What is the annual fee?",
        ],
    },
    {
        "name": "Scenario 2: Customer Checking Account & Transaction History (Voice Call)",
        "phone": "01898765432",
        "turns": [
            "Hi, I need to check my account balance",
            "Can you also show me my recent transactions?",
            "What is the minimum balance I need to maintain?",
            "Thanks, that's all I needed",
        ],
    },
    {
        "name": "Scenario 3: Frustrated Customer With Missing Salary Slip (Web Chat)",
        "phone": "01712345678",
        "turns": [
            "I applied for a loan but you asked for salary certificate",
            "I don't have a salary certificate, I'm a freelancer",
            "What alternative documents can I submit instead?",
            "How long does loan approval take?",
            "This is taking too long, can I speak to someone?",
        ],
    },
    {
        "name": "Scenario 4: Customer Wants to Block Lost Card (Voice Call - Urgent)",
        "phone": "01551234567",
        "turns": [
            "I lost my debit card, please block it immediately",
            "It's my debit card",
            "I think someone might have stolen it at the bazaar",
            "Will I get a replacement card?",
            "How long will the replacement take?",
        ],
    },
    {
        "name": "Scenario 5: Digital Banking Help - Password Reset & Limits (Web Chat)",
        "phone": "01898765432",
        "turns": [
            "I forgot my internet banking password",
            "How do I reset it?",
            "What is the daily transfer limit for internet banking?",
            "Can I increase the limit?",
            "Also whats my current balance?",
        ],
    },
]


def main():
    system = build_system()

    for scenario in SCENARIOS:
        run_scenario(system, scenario["name"], scenario["phone"], scenario["turns"])
        print()

    print("\n" + "="*70)
    print("  ALL 5 SCENARIOS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
