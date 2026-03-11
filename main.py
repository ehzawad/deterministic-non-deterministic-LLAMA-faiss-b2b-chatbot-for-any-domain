#!/usr/bin/env python3
"""CLI entry point for the B2B Banking Chatbot.

Usage:
    python main.py           # normal mode
    python main.py --debug   # show Pass 1 JSON, FAQ scores, state transitions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B2B Banking Chatbot")
    parser.add_argument("--debug", action="store_true", help="Show debug info per turn")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU-only mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  B2B Banking Chatbot — Dual-Model Architecture")
    print("  Qwen3-4B (Agentic) + Llama 3.1 8B (Generator)")
    print("=" * 60)
    print()

    # ── Phase 1: Load models ──────────────────────────────────────
    print("[1/5] Loading LLM models...")
    from engine.llm import DualLLMEngine

    agent_config = AgentModelConfig()
    generator_config = GeneratorModelConfig()

    if args.no_gpu:
        agent_config = AgentModelConfig(N_GPU_LAYERS=0)
        generator_config = GeneratorModelConfig(N_GPU_LAYERS=0)

    llm_engine = DualLLMEngine(agent_config, generator_config)
    print("       Qwen3-4B (agent) + Llama 3.1 8B (generator) loaded.\n")

    # ── Phase 2: Load FAQ system ──────────────────────────────────
    print("[2/5] Loading FAQ engine...")
    from faq.embedder import Embedder
    from faq.faq_engine import FAQEngine

    embedding_config = EmbeddingConfig()
    faq_config = FAQConfig()

    embedder = Embedder(
        model_name=embedding_config.MODEL_NAME,
        device=embedding_config.DEVICE if not args.no_gpu else "cpu",
    )
    faq_engine = FAQEngine(embedder, faq_config)
    print(f"       {faq_engine.index.ntotal} FAQ entries indexed.\n")

    # ── Phase 3: Initialize memory stores ─────────────────────────
    print("[3/5] Initializing memory stores...")
    from memory.session_store import SessionStore
    from memory.fact_store import FactStore
    from memory.thread_tracker import ThreadTracker
    from memory.emotion_tracker import EmotionTracker
    from memory.turn_history import TurnHistory

    memory_config = MemoryConfig()
    session_store = SessionStore()
    fact_store = FactStore()
    thread_tracker = ThreadTracker()
    emotion_tracker = EmotionTracker()
    turn_history = TurnHistory(maxlen=memory_config.MAX_TURN_HISTORY)
    print("       5 memory stores ready.\n")

    # ── Phase 4: Initialize tools, workflow, state machine ────────
    print("[4/5] Initializing tools, workflows, and state machine...")
    from tools.tool_registry import TOOL_REGISTRY
    from tools import mock_tools
    from tools.tool_executor import ToolExecutor
    from workflow.workflow_engine import WorkflowEngine
    from workflow.workflow_definitions import WORKFLOW_REGISTRY
    from state.dialogue_state_machine import DialogueStateMachine
    from state.auth_gate import AuthGate
    from guardrails.guardrails import GuardrailsPipeline

    tool_executor = ToolExecutor(TOOL_REGISTRY, mock_tools)
    workflow_engine = WorkflowEngine()
    state_machine = DialogueStateMachine()
    auth_gate = AuthGate(tool_executor, session_store, state_machine, max_attempts=AuthConfig().MAX_AUTH_ATTEMPTS)
    guardrails = GuardrailsPipeline(fact_store, emotion_tracker)
    print("       All subsystems initialized.\n")

    # ── Phase 5: Build the brain ──────────────────────────────────
    print("[5/5] Assembling the brain...")
    from engine.context_builder import ContextBuilder
    from engine.pass1_resolver import Pass1Resolver
    from engine.pass2_naturalizer import Pass2Naturalizer
    from engine.action_router import ActionRouter

    # Build tool definition list for context builder
    tool_defs = []
    for name, tool_def in TOOL_REGISTRY.items():
        tool_defs.append({
            "name": name,
            "description": tool_def.description,
            "parameters": tool_def.params_schema,
        })

    context_builder = ContextBuilder(llm_engine, tool_defs)
    pass1 = Pass1Resolver(llm_engine, context_builder)
    pass2 = Pass2Naturalizer(llm_engine, context_builder)

    action_router = ActionRouter(
        pass1=pass1,
        pass2=pass2,
        faq_engine=faq_engine,
        tool_executor=tool_executor,
        workflow_engine=workflow_engine,
        guardrails=guardrails,
        state_machine=state_machine,
        session_store=session_store,
        fact_store=fact_store,
        thread_tracker=thread_tracker,
        emotion_tracker=emotion_tracker,
        turn_history=turn_history,
    )
    print("       Brain assembled. Ready to chat!\n")

    # ── Authentication phase ──────────────────────────────────────
    print("-" * 60)
    greeting = auth_gate.start_auth()
    print(f"Bot: {greeting}")

    while not auth_gate.is_verified and not auth_gate.is_locked:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not user_input:
            continue

        # Record user turn
        turn_number = turn_history.add(role="user", text=user_input)

        success, message = auth_gate.attempt_verify(
            phone=user_input,
            fact_store=fact_store,
            turn_number=turn_number,
        )
        print(f"Bot: {message}")

        # Record bot turn
        turn_history.add(role="assistant", text=message)

        if auth_gate.is_locked:
            print("\n[Session ended — transferred to human agent]")
            return

    # Transition to listening mode
    state_machine.transition("start_listening")
    state_machine.transition("ready")

    # ── Main conversation loop ────────────────────────────────────
    print("-" * 60)
    print("(Type 'quit' or 'bye' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            state_machine.transition("goodbye")
            print("Bot: Thank you for banking with us. Have a great day!")
            break

        # Process the turn through the full pipeline
        result = action_router.process_turn(user_input)

        # Show filler message if present
        if result.filler_message:
            print(f"Bot: {result.filler_message}")

        # Show final response
        print(f"Bot: {result.final_response}")

        # Debug output
        if args.debug:
            print(f"  [DEBUG] action_type: {result.action_type}")
            print(f"  [DEBUG] state: {state_machine.current_state}")
            print(f"  [DEBUG] emotion: {emotion_tracker.latest}")
            if result.debug:
                for key, value in result.debug.items():
                    val_str = str(value)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    print(f"  [DEBUG] {key}: {val_str}")
            print()

        # Check for terminal states
        if state_machine.is_terminal:
            if state_machine.current_state.name == "HUMAN_TRANSFER":
                print("\n[Session ended — transferred to human agent]")
            else:
                print("\n[Session ended]")
            break

    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Turns: {turn_history.turn_counter}")
    print(f"  Final state: {state_machine.current_state.name}")
    print(f"  Emotion trajectory: {emotion_tracker.to_context_string()}")
    if fact_store.all_latest():
        print(f"  Facts: {fact_store.to_context_string()}")
    print()


if __name__ == "__main__":
    main()
