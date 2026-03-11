"""ContextBuilder -- assembles per-model prompts from the five memory stores.

Pass 1 (Qwen3):  full agentic routing prompt with tool definitions, all
                  memory contexts, few-shot examples, multi-turn reference
                  instructions, and a strict JSON output instruction.

Pass 2 (Llama):  tier-aware naturalization prompt that converts structured
                  data into conversational language with per-tier guidance.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from engine.llm import DualLLMEngine
from memory.session_store import SessionStore
from memory.fact_store import FactStore
from memory.thread_tracker import ThreadTracker
from memory.emotion_tracker import EmotionTracker
from memory.turn_history import TurnHistory


# -- JSON schema that Pass 1 must produce --------------------------------
_PASS1_OUTPUT_SCHEMA: dict[str, Any] = {
    "action_type": "faq|tool_call|talk|escalate",
    "resolved_query": "string or null -- rewritten query for FAQ/tool",
    "tool_calls": [
        {
            "function": "tool_name",
            "params": {"param": "value"},
        }
    ],
    "workflow_command": {
        "action": "start|continue|cancel",
        "workflow": "workflow_name",
        "data": {},
    },
    "talk_subtype": "greeting|empathy|clarification|proactive or null",
    "direct_response": "string or null -- only for talk actions",
    "emotion_read": "neutral|concerned|frustrated|anxious|angry|stressed",
    "filler_message": "string or null -- brief acknowledgement while processing",
}

# -- Few-shot examples for Pass 1 ----------------------------------------
_FEW_SHOT_EXAMPLES: str = textwrap.dedent("""\
    [Few-Shot Examples]
    These show EXACTLY how to format output. Follow these patterns.

    User: "Am I eligible for a credit card?"
    {"action_type":"faq","resolved_query":"credit card eligibility criteria","tool_calls":null,"workflow_command":null,"talk_subtype":null,"direct_response":null,"emotion_read":"neutral","filler_message":"Let me check that for you."}

    User: "What documents do I need?"
    {"action_type":"faq","resolved_query":"required documents for credit card application","tool_calls":null,"workflow_command":null,"talk_subtype":null,"direct_response":null,"emotion_read":"neutral","filler_message":"Let me look that up."}

    User: "How do I reset my internet banking password?"
    {"action_type":"faq","resolved_query":"internet banking password reset process","tool_calls":null,"workflow_command":null,"talk_subtype":null,"direct_response":null,"emotion_read":"neutral","filler_message":"Let me check that for you."}

    User: "What's my balance?"
    {"action_type":"tool_call","resolved_query":null,"tool_calls":[{"function":"get_balance","params":{"account_type":"savings"}}],"workflow_command":null,"talk_subtype":null,"direct_response":null,"emotion_read":"neutral","filler_message":"Checking your balance now."}

    User: "Show me my recent transactions"
    {"action_type":"tool_call","resolved_query":null,"tool_calls":[{"function":"get_transaction_history","params":{"account_type":"current","days":30}}],"workflow_command":null,"talk_subtype":null,"direct_response":null,"emotion_read":"neutral","filler_message":"Retrieving your transactions."}

    User: "Check my credit score"
    {"action_type":"tool_call","resolved_query":null,"tool_calls":[{"function":"get_credit_score","params":{}}],"workflow_command":null,"talk_subtype":null,"direct_response":null,"emotion_read":"neutral","filler_message":"Checking your credit score."}

    User: "Thanks, bye!"
    {"action_type":"talk","resolved_query":null,"tool_calls":null,"workflow_command":null,"talk_subtype":"greeting","direct_response":"Goodbye! Have a great day.","emotion_read":"neutral","filler_message":null}

    User: "I want to speak to a manager"
    {"action_type":"escalate","resolved_query":null,"tool_calls":null,"workflow_command":null,"talk_subtype":null,"direct_response":null,"emotion_read":"frustrated","filler_message":"Let me connect you with a specialist."}

    Multi-turn example:
    History: [user asked about credit cards, assistant gave eligibility info]
    User: "Is that enough to get approved?"
    {"action_type":"faq","resolved_query":"credit card approval minimum requirements","tool_calls":null,"workflow_command":null,"talk_subtype":null,"direct_response":null,"emotion_read":"neutral","filler_message":"Let me check that."}""")


class ContextBuilder:
    """Builds prompts for the two-pass pipeline."""

    def __init__(
        self,
        llm_engine: DualLLMEngine,
        tool_definitions: list[dict[str, Any]],
    ) -> None:
        self._llm = llm_engine
        self._tool_defs = tool_definitions
        self._tool_defs_json = json.dumps(tool_definitions, indent=2)
        self._pass1_schema_json = json.dumps(_PASS1_OUTPUT_SCHEMA, indent=2)

    # ------------------------------------------------------------------
    # Pass 1: Qwen3 agentic routing prompt
    # ------------------------------------------------------------------
    def build_pass1_prompt(
        self,
        session_store: SessionStore,
        fact_store: FactStore,
        thread_tracker: ThreadTracker,
        emotion_tracker: EmotionTracker,
        turn_history: TurnHistory,
        current_utterance: str,
    ) -> tuple[str, str]:
        """Assemble system prompt and user message for Qwen3 chat completion.

        Returns (system_prompt, user_message) for create_chat_completion.
        """

        sections: list[str] = []

        # ---- system ----
        sections.append(textwrap.dedent("""\
            [System]
            You are an agentic router for a banking chatbot. Analyze the customer's
            utterance and decide which action to take. Output ONLY valid JSON.

            ROUTING RULES (follow strictly):

            "tool_call" — USE THIS when customer wants THEIR OWN data or to DO something:
              Keywords: balance, my balance, my transactions, my credit score, block card, file dispute
              Available tools: get_balance, get_transaction_history, get_credit_score, block_card, file_dispute
              ALWAYS use tool_call for: "whats my balance", "check my balance", "show my transactions",
              "block my card", "check my credit score", "file a dispute"

            "faq" — USE THIS for questions about bank POLICIES, RULES, PROCESSES:
              Keywords: eligible, eligibility, documents, how long, fees, annual fee, limits, minimum balance,
              requirements, how to, what is the process, password reset, interest rate, opening hours
              ALWAYS use faq for: "am I eligible", "what documents", "how to reset password",
              "what are the fees", "what is the minimum balance", "how long does it take"

            "talk" — ONLY for greetings, thank-you, goodbye, or when customer says something
              that is neither a question nor a request. NEVER use talk for questions about banking.

            "escalate" — Customer explicitly demands human agent OR is very angry/frustrated.

            CRITICAL: If the utterance is a QUESTION about banking, it is NEVER "talk".
            Questions = faq or tool_call. Only pleasantries = talk.

            MULTI-TURN REFERENCE RESOLUTION:
            Read [Turn History] carefully. If the user refers to something from earlier
            (e.g., "that card", "is that enough", "the one you mentioned", "how about that"),
            resolve the reference from history. Rewrite resolved_query to include the full
            context so the FAQ or tool system can understand it without seeing the history."""))

        # ---- few-shot examples ----
        sections.append(_FEW_SHOT_EXAMPLES)

        # ---- tool definitions ----
        sections.append(f"[Tool Definitions]\n{self._tool_defs_json}")

        # ---- output schema ----
        sections.append(
            f"[Output Schema]\n{self._pass1_schema_json}\n"
            f"For tool_calls, use keys \"function\" and \"params\"."
        )

        # System prompt is everything above
        system_prompt = "\n\n".join(sections)

        # User message = memory context + utterance
        user_parts: list[str] = []
        user_parts.append(f"[Session] {session_store.to_context_string()}")
        user_parts.append(f"[Facts] {fact_store.to_context_string()}")
        user_parts.append(f"[Threads] {thread_tracker.to_context_string()}")
        user_parts.append(f"[Emotion] {emotion_tracker.to_context_string()}")
        user_parts.append(f"[Turn History]\n{turn_history.to_context_string()}")
        user_parts.append(f"[Customer says] {current_utterance}")
        user_parts.append("/no_think\nRespond with ONLY a JSON object. No other text.")

        user_message = "\n".join(user_parts)

        return system_prompt, user_message

    # ------------------------------------------------------------------
    # Pass 2: Llama naturalization prompt
    # ------------------------------------------------------------------
    def build_pass2_prompt(
        self,
        pass1_result: dict[str, Any],
        ground_truth: str | None,
        faq_result: Any | None = None,
        session_store: SessionStore | None = None,
        emotion_state: str = "neutral",
        hedged: bool = False,
    ) -> str:
        """Assemble the prompt sent to the Llama generator model."""

        sections: list[str] = []

        # ---- system ----
        sections.append(textwrap.dedent("""\
            [System]
            You are a friendly, professional banking assistant. Your task is to
            take structured data and present it naturally in conversation.
            Be concise, warm, and clear. Do NOT invent information -- only use
            what is provided below.

            You are speaking DIRECTLY to the customer in a live banking interaction.
            Never break character. Never add meta-commentary like "How was that?"
            or "Is there anything else?" or "Let me know if you need anything".
            Just answer the question concisely."""))

        # ---- customer context ----
        if session_store is not None and session_store.is_sealed:
            sections.append(
                f"[Customer Context]\n"
                f"Name: {session_store.name}\n"
                f"Account type: {session_store.account_type}\n"
                f"Current emotion: {emotion_state}"
            )
        else:
            sections.append(
                f"[Customer Context]\n"
                f"Unauthenticated customer\n"
                f"Current emotion: {emotion_state}"
            )

        # ---- ground truth data ----
        if ground_truth:
            sections.append(f"[Ground Truth]\n{ground_truth}")
        elif faq_result is not None:
            sections.append(f"[FAQ Result]\n{faq_result}")

        # ---- action context and tier-specific instructions ----
        action = pass1_result.get("action_type", "talk")
        sections.append(f"[Action Type]\n{action}")

        # ---- tier-specific guidance ----
        tier_instruction = self._build_tier_instruction(
            action, faq_result, hedged,
        )
        sections.append(f"[Tier Guidance]\n{tier_instruction}")

        # ---- instruction ----
        sections.append(textwrap.dedent("""\
            [Instruction]
            Write a single conversational reply to the customer. Rules:
            - Do NOT use markdown, bullet points, or headers
            - Keep it to 1-3 short sentences unless the data requires more
            - Do NOT add meta-commentary like "How was that?" or "Let me know"
            - Do NOT break character or mention you are an AI
            - Do NOT end with questions like "Is there anything else I can help with?"
            - Just give the answer naturally as a real bank employee would
            - Start your reply directly with the answer"""))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Tier-specific Pass 2 instruction builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tier_instruction(
        action: str,
        faq_result: Any | None,
        hedged: bool,
    ) -> str:
        """Return tier-specific naturalisation guidance for Pass 2."""

        # Determine the FAQ tier from the faq_result object
        tier: str | None = None
        if faq_result is not None and hasattr(faq_result, "tier"):
            tier = faq_result.tier

        if action == "tool_call":
            return (
                "Present this tool result data in ONE short sentence. "
                "Example: 'Your current account balance is 45,200 BDT.' "
                "Do NOT elaborate, do NOT offer investment advice, "
                "do NOT list transactions unless asked. Just state the key number."
            )

        if action == "faq":
            if tier == "PURE_FAQ":
                return (
                    "Present this information as fact. Just wrap it naturally "
                    "in 1-2 sentences. Do not add hedging language -- the FAQ "
                    "match is high-confidence."
                )
            if tier == "BLENDED":
                return (
                    "Use this FAQ answer as your BASE. Reason ON TOP of it. "
                    "Use hedge words: typically, usually, in most cases. "
                    "Never say definitely, always, guaranteed."
                )
            if tier == "PURE_LLM":
                return (
                    "No reliable FAQ match found. Generate a helpful response "
                    "but add a disclaimer suggesting the customer verify at a "
                    "branch or by calling customer service for the most "
                    "accurate information."
                )
            # Fallback for FAQ without a recognized tier
            if hedged:
                return (
                    "The answer confidence is MEDIUM. Hedge appropriately -- "
                    "use phrases like \"Based on what I can see...\" or "
                    "\"It appears that...\" and suggest the customer verify "
                    "with a branch or specialist if needed."
                )
            return "High confidence. Present the information directly."

        if action == "escalate":
            return (
                "The customer is being transferred to a human agent. "
                "Acknowledge their concern empathetically and let them know "
                "help is on the way."
            )

        if action == "talk":
            return (
                "This is a conversational response. Be warm and natural. "
                "Respond directly without unnecessary elaboration."
            )

        # Default fallback
        return "Present the information directly and concisely."
