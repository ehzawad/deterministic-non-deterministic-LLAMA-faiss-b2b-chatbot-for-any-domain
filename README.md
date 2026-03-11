# Deterministic + Non-Deterministic B2B Chatbot

A domain-agnostic NLP pipeline for B2B customer service, built on a dual-model architecture with quantized local LLMs and FAISS-backed retrieval. Designed from a 10-part architectural specification covering the full conversation lifecycle.

The system handles both deterministic responses (fixed FAQ answers from a knowledge base) and non-deterministic responses (LLM-generated reasoning) within a single conversation flow, with guardrails to prevent hallucination and ensure factual accuracy.

## Architecture

```
User utterance
     |
     v
[Context Builder] -- assembles prompt from 5 memory stores
     |
     v
[Qwen3-4B] -- Pass 1: intent resolution, action routing, structured JSON
     |
     +--- faq ---------> [FAQ Engine] -- FAISS similarity search on knowledge base
     |                        |
     +--- tool_call ----> [Tool Executor] -- execute API calls, write to fact store
     |                        |
     +--- talk ---------> [Direct Response]
     |                        |
     +--- workflow -----> [Workflow Engine] -- multi-step flows with suspend/resume
     |                        |
     +--- escalate -----> [Human Transfer]
     |                        |
     v                        v
[Llama 3.1 8B] -- Pass 2: naturalize ground truth into conversational response
     |
     v
[Guardrails] -- G1: hallucination blocker, G4: hedge enforcer, G5: emotion escalation
     |
     v
Final Response --> Update all 5 memory stores --> next turn
```

### Dual-Model Design

The pipeline uses two specialized models instead of one:

- **Qwen3-4B (Q8_0)** -- Agentic brain. Handles intent classification, action routing, tool calling decisions, and emotion detection. Outputs structured JSON via chat completion. Chosen for its native tool-calling capability and thinking/non-thinking modes.

- **Llama 3.1 8B Instruct (Q4_K_M)** -- Language brain. Takes ground truth data (FAQ answers, tool results) and produces natural conversational responses. Handles tone adjustment based on customer emotion state and FAQ confidence tier.

Both models run simultaneously on a single GPU via llama-cpp-python with CUDA. The system auto-detects available VRAM and computes optimal context window sizes, with automatic retry at smaller context if memory is tight.

### FAQ Engine (Three Response Modes)

The FAQ engine implements a three-tier confidence routing system using FAISS vector similarity search with E5 multilingual embeddings:

- **PURE FAQ** (similarity > 0.85) -- Returns the fixed knowledge base answer directly. No LLM generation needed. Deterministic and fast.
- **BLENDED** (0.60 - 0.85) -- Uses the FAQ answer as a base, then the LLM reasons on top of it with hedge language ("typically", "usually"). Combines deterministic data with non-deterministic reasoning.
- **PURE LLM** (< 0.60) -- No reliable FAQ match. The LLM generates freely with a disclaimer to verify at a branch.

### Memory System (Five Stores)

Every conversation turn reads from and writes to five memory stores:

1. **Session Store** -- Written once at authentication. Customer name, account type, verification status.
2. **Fact Store** -- Written only by tool results, never by the LLM. Versioned (corrections append, never overwrite). Source of truth for account data.
3. **Thread Tracker** -- Tracks active conversation topics with status (ACTIVE/SUSPENDED/RESOLVED) and slot states.
4. **Emotion Tracker** -- Per-turn emotion trajectory (neutral, concerned, frustrated, anxious, angry). Drives tone adjustment and escalation triggers.
5. **Turn History** -- Raw verbatim ring buffer of recent turns including tool calls and results.

### Guardrails

Five safety layers run on every response:

- **G1 -- Hallucination Blocker**: Regex scans for financial data (currency amounts, account numbers, percentages) in LLM output. If found without a prior tool call, the response is blocked and the system prompts a tool lookup instead. Bypassed for FAQ-sourced answers.
- **G2 -- Tool Call Validator**: Validates tool call schemas before execution. Invalid calls get one retry, then a template fallback.
- **G3 -- Fact Integrity**: Architectural enforcement -- only the tool executor writes to the fact store. No code path allows LLM output to become a stored fact.
- **G4 -- Hedge Enforcer**: When FAQ confidence is medium (BLENDED tier), scans for forbidden absolute language ("definitely", "always", "guaranteed") and replaces with hedges ("typically", "usually").
- **G5 -- Emotion Escalation**: If customer emotion has been rising for 3+ consecutive turns, proactively offers human agent transfer.

### Dialogue State Machine

A 26-state finite state machine tracks conversation phase:

```
Greeting -> Authenticating -> Verified -> Listening/Ready
  |
  +-- FAQ Lookup -> PureFAQ / BlendedFAQ / LLMFallback -> Ready
  +-- Tool Action -> Executing / Chaining -> ToolDone -> Ready
  +-- Workflow Active -> StepExec / WaitUser -> Done / Suspended
  +-- Farewell
  +-- Human Transfer
```

## Setup

Requirements: Python 3.12, NVIDIA GPU with CUDA (T4 or better), ~9GB disk for models.

```bash
git clone git@github.com:ehzawad/deterministic-non-deterministic-LLAMA-faiss-b2b-chatbot-for-any-domain.git
cd deterministic-non-deterministic-LLAMA-faiss-b2b-chatbot-for-any-domain

python3.12 -m venv .venv
source .venv/bin/activate

# Install with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install sentence-transformers faiss-cpu numpy jsonschema huggingface-hub

# Download models (~9.2GB)
python setup_model.py

# Build FAQ index
python build_faq_index.py
```

### GPU Memory

The system auto-detects VRAM and allocates context windows accordingly:

| GPU | VRAM | Agent ctx | Generator ctx |
|-----|------|-----------|---------------|
| T4 | 15 GB | ~20K | ~5K (auto-retry) |
| L4 | 24 GB | 40K (native max) | ~66K |
| L40 | 48 GB | 40K (native max) | 131K (native max) |
| A100 | 80 GB | 40K (native max) | 131K (native max) |

## Usage

```bash
# Interactive chat
python main.py

# With debug output (shows action routing, FAQ confidence, state transitions)
python main.py --debug
```

### Example Conversation

```
Bot: Welcome! Can I get your phone number?
You: 01712345678
Bot: Welcome, Rafiqul Islam!

You: Am I eligible for a credit card?
Bot: To apply for a credit card you must be a Bangladeshi citizen or resident
     aged 21-65 with a minimum monthly income of BDT 25,000...
     [action=faq, confidence=0.95]

You: what documents do I need?
Bot: You will need: (1) completed application form, (2) photocopy of NID,
     (3) two passport-size photographs, (4) last 3 months bank statements...
     [action=faq, confidence=0.94]

You: I dont have salary slip
Bot: If a salary certificate is unavailable you may submit bank statements
     showing regular salary credits for the last 6 months, an employer-issued
     income verification letter, or your latest income tax return...
     [action=faq, confidence=0.92]

You: whats my balance?
Bot: Your current account balance is 87,345 BDT.
     [action=tool_call, tool=get_balance]

You: Thanks, goodbye
Bot: Goodbye! Have a great day.
     [action=talk]
```

## Testing

```bash
# Run the D10 reference conversation (from the architecture spec)
python tests/test_full_conversation.py

# Run 5 realistic call center / web chat scenarios
python tests/test_realistic_scenarios.py
```

The five test scenarios cover:
1. New customer credit card inquiry (web chat)
2. Account balance and transaction history (voice call)
3. Frustrated customer with missing salary slip, escalation to human (web chat)
4. Lost/stolen card blocking (voice call, urgent)
5. Digital banking password reset and transfer limits (web chat)

## Project Structure

```
docs/architecture/          10 PDF specification documents (D1-D10)

engine/
  llm.py                   Dual-model engine (Qwen3 + Llama, GPU-aware)
  context_builder.py        7-layer prompt assembly from memory stores
  pass1_resolver.py         Qwen3 structured JSON output (intent + routing)
  pass2_naturalizer.py      Llama naturalization (ground truth -> conversation)
  action_router.py          Main loop orchestrator (D2 spec implementation)

memory/
  session_store.py          Write-once auth data
  fact_store.py             Versioned tool-only facts
  thread_tracker.py         Topic tracking with slot states
  emotion_tracker.py        Per-turn emotion trajectory
  turn_history.py           Raw verbatim ring buffer

faq/
  embedder.py               E5-multilingual-large-instruct wrapper
  faq_engine.py             FAISS search with three-tier routing

tools/
  tool_registry.py          Tool definitions with JSON schemas
  mock_tools.py             Mock banking API implementations
  tool_executor.py          Schema validation, execution, fact writing

workflow/
  workflow_engine.py         Multi-step flow execution with suspend/resume
  workflow_definitions.py    Card blocking, dispute filing flows

state/
  dialogue_state_machine.py  26-state FSM
  auth_gate.py               Phone verification with 3x lockout

guardrails/
  guardrails.py              Pipeline runner (G1 -> G4 -> G5)
  hallucination_blocker.py   G1: financial data without tool source
  hedge_enforcer.py          G4: medium-confidence hedge language
  emotion_escalation.py      G5: rising emotion -> human transfer offer

config.py                    GPU-aware auto-configuration
setup_model.py               Model downloader
build_faq_index.py           FAQ knowledge base + FAISS index builder
main.py                      CLI entry point
```

## Adapting to Other Domains

The system is domain-agnostic. To adapt for a different business:

1. Replace `faq/knowledge_base.json` with your domain FAQ entries
2. Replace `tools/mock_tools.py` with your actual API integrations
3. Update `tools/tool_registry.py` with your tool schemas
4. Update `workflow/workflow_definitions.py` with your multi-step flows
5. Rebuild the FAISS index: `python build_faq_index.py`

The core pipeline (dual-model routing, memory system, guardrails, state machine) remains unchanged.

## Architecture Specification

The 10 PDF documents in `docs/architecture/` define the complete system design:

| Doc | Title | Covers |
|-----|-------|--------|
| D1 | NLP Engine Overview | Top-level data flow: Auth Gate, LLM backbone, Action Layer, Response Assembly, Memory Layer, Context Builder |
| D2 | The Main Loop - One Turn | Single conversation turn lifecycle from raw utterance to final response |
| D3 | LLM Decision Space | Action type routing: faq, tool, workflow, talk, escalate |
| D4 | Memory System - 5 Stores | Session, facts, threads, emotion, turn history |
| D5 | Context Window Assembly | 7-layer prompt construction from memory stores |
| D6 | FAQ Engine - Three Response Modes | Pure FAQ, blended, and pure LLM response strategies |
| D7 | Tool Executor + Workflow Engine | API execution, tool chaining, multi-step workflows |
| D8 | Dialogue State Machine | 26-state FSM with full transition table |
| D9 | Guardrails - 5 Safety Layers | Hallucination blocking, fact integrity, hedge enforcement, emotion escalation |
| D10 | Real Conversation | End-to-end sequence diagram showing all systems in a banking scenario |
