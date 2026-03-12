# System Architecture

This document describes the implemented NLP pipeline as built, not the original PDF specifications. The original 10 PDF documents in `docs/architecture/` served as the design blueprint. This document reflects the actual running system.

## Overview

A dual-model conversational pipeline that splits responsibilities between an agentic router (Qwen3-4B) and a language generator (Llama 3.1 8B). The router classifies intent and produces structured JSON. The generator naturalizes ground truth data into conversational responses. Five memory stores maintain conversation state across turns.

```
User utterance
     |
     v
Context Builder (assembles from 5 memory stores)
     |
     v
Qwen3-4B (Pass 1: structured JSON via chat completion)
  action_type: faq | tool_call | talk | workflow | escalate
  resolved_query, tool_calls, emotion_read, filler_message
     |
     +-- faq --------> FAQ Engine (FAISS + E5 embeddings)
     |                   |-- PURE_FAQ (>0.85): return answer directly
     |                   |-- BLENDED (0.6-0.85): Pass 2 + hedge words
     |                   |-- PURE_LLM (<0.6): Pass 2 generates freely
     |
     +-- tool_call --> Tool Executor (schema validate -> execute -> write facts)
     |
     +-- talk -------> Direct response from Pass 1 (skip Pass 2)
     |
     +-- workflow ---> Workflow Engine (multi-step with suspend/resume)
     |
     +-- escalate --> Human transfer (suspend all threads)
     |
     v
Llama 3.1 8B (Pass 2: naturalize ground truth)
     |
     v
Guardrails (G1: hallucination blocker, G4: hedge enforcer, G5: emotion escalation)
     |
     v
Final Response -> Update memory stores -> next turn
```

## Models

### Agent: Qwen3-4B (Q8_0, 4.3 GB)

Handles intent classification, action routing, tool calling decisions, and emotion detection. Uses chat completion format (system + user messages) for reliable structured JSON output. Supports thinking/non-thinking mode (`/no_think` for fast routing).

- Context window: auto-sized per GPU (up to 40,960 native max)
- Temperature: 0.6, Top-P: 0.95, Top-K: 20
- Presence penalty: 1.5 (suppresses repetition in quantized models)
- Output: structured JSON parsed with brace-depth tracking, `<think>` tags stripped

### Generator: Llama 3.1 8B Instruct (Q4_K_M, 4.9 GB)

Takes ground truth data and produces natural conversational responses. Tier-specific prompting controls output style based on FAQ confidence or action type.

- Context window: auto-sized per GPU (up to 131,072 native max)
- Temperature: 0.7, Top-P: 0.9
- Max tokens: 200 (concise banking responses)
- Stop tokens: structural boundaries only (`\n\n\n`, triple backtick, `</s>`, conversation markers)
- Post-processing: strips prompt leakage artifacts and trailing noise

### Embeddings: E5-multilingual-large-instruct (1024-dim)

Runs on CPU to preserve GPU VRAM for the two LLMs.

- Query prefix: `"Instruct: Retrieve the most relevant banking FAQ answer\nQuery: {query}"`
- Document prefix: `"passage: {text}"`
- All vectors L2-normalized for cosine similarity via inner product

### GPU Memory Management

The system auto-detects available VRAM at startup and computes optimal context windows. Both LLMs load simultaneously with all layers offloaded to GPU (`n_gpu_layers=-1`). If a model fails to allocate context, it retries with halved n_ctx until it fits.

Fixed VRAM cost: ~10.8 GB (model weights + CUDA overhead)
KV cache cost: Qwen3 ~90 KB/token, Llama ~128 KB/token
Remaining VRAM split: 40% agent, 60% generator

| GPU | Agent n_ctx | Generator n_ctx |
|-----|-------------|-----------------|
| T4 (15 GB) | ~20K | ~5K |
| L4 (24 GB) | 40K | ~66K |
| L40 (48 GB) | 40K | 131K |
| A100 (80 GB) | 40K | 131K |

## Two-Pass Pipeline

### Pass 1: Intent Resolution (Qwen3)

The system prompt contains:
1. Routing rules with explicit keyword-to-action mappings
2. Nine few-shot JSON examples covering all action types plus multi-turn references
3. Tool definitions with JSON schemas for all 6 registered tools
4. The output JSON schema

The user message contains:
1. Session context (customer name, account type, verification status)
2. Fact store snapshot (all tool-produced facts with turn numbers)
3. Thread states (active/suspended/resolved topics)
4. Emotion trajectory (with escalation flag if rising)
5. Turn history (last 16 turns verbatim including tool calls and results)
6. Current utterance
7. `/no_think` control signal

Output JSON schema:
```json
{
  "action_type": "faq|tool_call|talk|escalate|workflow",
  "resolved_query": "cleaned search query for FAQ",
  "tool_calls": [{"function": "tool_name", "params": {}}],
  "workflow_command": {"action": "start|continue|cancel", "workflow": "name"},
  "talk_subtype": "greeting|empathy|clarification|proactive",
  "direct_response": "text for talk actions",
  "emotion_read": "neutral|concerned|frustrated|anxious|angry|stressed",
  "filler_message": "acknowledgement streamed while processing"
}
```

Fallback: if JSON parsing fails twice, defaults to `{"action_type": "talk", "direct_response": <raw output>}`.

### Pass 2: Naturalization (Llama 3.1)

Tier-specific prompting based on the action type and FAQ confidence:

- **tool_call**: "Present in ONE short sentence. Do NOT elaborate."
- **faq/PURE_FAQ**: "Present as fact. 1-2 sentences. No hedging."
- **faq/BLENDED**: "Use FAQ as BASE. Reason ON TOP. Hedge with: typically, usually."
- **faq/PURE_LLM**: "No FAQ match. Generate helpfully but add disclaimer."
- **talk**: Pass 2 skipped entirely, direct_response from Pass 1 used
- **escalate**: Fixed escalation message, no LLM call

## FAQ Engine

### Index

- Type: FAISS IndexFlatIP (inner product on unit vectors = cosine similarity)
- Dimension: 1024 (E5-large-instruct)
- Entries: 30 across 7 categories
- Categories: Credit Cards (8), Accounts (6), Loans (5), Digital Banking (4), Disputes (4), General (3)

### Three-Tier Routing

```
Resolved query from Pass 1 (NOT raw utterance)
     |
     v
Embed with E5 (1024-dim, normalized)
     |
     v
FAISS search top-3
     |
     +-- similarity > 0.85 --> PURE FAQ
     |     Return fixed answer. No Pass 2 call. Deterministic.
     |
     +-- 0.60 to 0.85 ------> BLENDED
     |     Answer as base context. Pass 2 reasons on top with hedges.
     |
     +-- < 0.60 ------------> PURE LLM
           No match. Pass 2 generates freely with disclaimer.
```

## Memory System

### Store 1: Session (write-once)

Written once during authentication, then sealed. Contains: name, account_type, verified flag, tenure, phone. Cannot be modified after sealing.

### Store 2: Facts (tool-only, versioned)

Written exclusively by the tool executor. The LLM never writes facts. Each fact is appended (corrections don't overwrite). Fields: key, value, turn_number, timestamp, source_tool.

Example: `balance=87345.5 @turn5 (source: get_balance)`

### Store 3: Threads (topic tracking)

Tracks active conversation topics with status (ACTIVE, SUSPENDED, RESOLVED) and slot states (filled/needed). The action router opens threads on FAQ and tool actions, resolves them after successful responses, and suspends all on escalation.

### Store 4: Emotion (trajectory)

Per-turn emotion recording. Valid levels (ordered): neutral, concerned, frustrated, anxious, angry, stressed. The `is_escalating(window=3)` method checks if the last 3 emotions form a strictly increasing sequence.

### Store 5: Turn History (ring buffer)

Last 16 turns stored verbatim in a `collections.deque`. Each turn includes: turn_number, role (user/assistant/tool), text, tool_calls list, tool_results list. Old turns silently evicted when buffer is full.

## Tools

Six registered tools with JSON Schema validation:

| Tool | Parameters | Returns |
|------|-----------|---------|
| verify_phone | phone (string) | Customer record (name, account_type, tenure, verified) |
| get_balance | account_type (savings/current) | balance (BDT), currency |
| get_transaction_history | account_type, days (default 30) | List of 5 recent transactions |
| get_credit_score | (none) | score (742), rating (Good), scale (300-900) |
| block_card | card_type (debit/credit), reason | status (blocked), reference (BLK-XXXXX) |
| file_dispute | transaction_id, reason | dispute_id (DSP-XXXX), estimated_resolution |

All tools are mock implementations with Bangladeshi banking data (BDT currency, bKash transfers, DESCO bills, Agora Supermarket, etc.). Replace with actual API integrations for production.

Tool results are automatically written to the Fact Store with key-value pairs.

## Workflows

Two multi-step workflows defined:

**card_block** (4 steps): Ask card type -> Ask reason -> Execute block_card -> Confirm with reference number

**dispute_flow** (4 steps): Ask transaction ID -> Ask reason -> Execute file_dispute -> Confirm with dispute ID

Each step has a type: `needs_input` (ask user), `auto` (execute tool), `proactive` (confirm result). Workflows can be suspended when the user switches topic and resumed later.

## Dialogue State Machine

24 states organized into 6 groups:

**Opening**: GREETING, AUTHENTICATING, COLLECT_ID, VERIFY, LOCKED, VERIFIED

**Core**: LISTENING, READY, CLARIFYING

**FAQ**: FAQ_SEARCHING, PURE_FAQ, BLENDED_FAQ, LLM_FALLBACK

**Tool**: EXECUTING, CHAINING, TOOL_DONE

**Workflow**: STEP_EXEC, WAIT_USER, WF_DONE, SUSPENDED

**Terminal**: HUMAN_TRANSFER, FAREWELL

The action router maps Pass 1 action types to state machine triggers:
- faq -> policy_question
- tool_call -> needs_data
- talk -> chit_chat
- workflow -> multi_step
- escalate -> demands_agent

After each action completes, an "answered" trigger returns the machine to READY.

## Guardrails

### G1: Hallucination Blocker

Regex scans LLM output for currency amounts, 10-16 digit account numbers, and percentages. If financial data is found without a corresponding tool call, the response is blocked and replaced with "I need to look that up to give you accurate information."

Bypassed when the response is FAQ-sourced (our own knowledge base, not hallucinated).

### G2: Tool Call Validator

Validates tool call function name and parameters against the tool registry JSON schemas before execution. Invalid calls get one retry, then a template fallback message.

### G3: Fact Integrity

Architectural enforcement. Only `ToolExecutor.execute()` calls `FactStore.add()`. No code path exists from LLM output to the fact store.

### G4: Hedge Enforcer

Active only for BLENDED FAQ tier (confidence 0.60-0.85). Scans response for forbidden absolute words ("definitely", "always", "guaranteed") and replaces them with hedge alternatives ("typically", "usually", "generally").

### G5: Emotion Escalation

Checks if customer emotion has been rising for 3+ consecutive turns. If detected, appends: "I can sense this is frustrating. Would you like me to connect you with a human agent who might be able to help more directly?"

## Authentication

Phone-based verification with 3-attempt lockout:

1. Customer provides phone number
2. System calls verify_phone tool against customer database
3. On success: seal session store, write facts, transition to VERIFIED
4. On failure: increment attempts, if 3x -> LOCKED -> HUMAN_TRANSFER
5. Locked state is terminal (requires human intervention)
