"""Central configuration for the B2B chatbot system.

GPU-aware: context windows are auto-calculated based on available VRAM
so the system uses maximum capacity on any GPU (T4, L4, L40, A100, etc.).
"""

from dataclasses import dataclass, field
import os


def _get_gpu_vram_mb() -> int:
    """Probe available GPU VRAM in MB. Returns 0 if no GPU."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0


def _compute_context_sizes(vram_mb: int) -> tuple[int, int]:
    """Compute optimal n_ctx for agent and generator given total VRAM.

    Both models must coexist in VRAM simultaneously.

    Model weights (fixed):
      Qwen3-4B Q8_0:          ~4280 MB
      Llama 3.1 8B Q4_K_M:    ~4920 MB
      E5-large-instruct fp16:  ~1100 MB
      CUDA overhead:            ~500 MB
      Total fixed:             ~10800 MB

    KV cache (scales with n_ctx) — measured from model architectures:
      Qwen3-4B:   36 layers * 8 KV heads * 80 dim * 2 (K+V) * 2 bytes = 92160 B/tok ≈ 90 KB/tok
      Llama 3.1:  32 layers * 8 KV heads * 128 dim * 2 (K+V) * 2 bytes = 131072 B/tok ≈ 128 KB/tok

    Strategy: allocate remaining VRAM to KV caches, 40% agent / 60% generator.
    Clamp to native max (Qwen3: 40960, Llama 3.1: 131072).
    """
    FIXED_MB = 10800
    QWEN3_KB_PER_TOKEN = 90
    LLAMA_KB_PER_TOKEN = 128
    QWEN3_MAX_CTX = 40960
    LLAMA_MAX_CTX = 131072

    if vram_mb <= 0:
        return 2048, 2048

    available_kb = max(vram_mb - FIXED_MB, 512) * 1024

    agent_budget_kb = int(available_kb * 0.4)
    gen_budget_kb = int(available_kb * 0.6)

    agent_ctx = min(agent_budget_kb // QWEN3_KB_PER_TOKEN, QWEN3_MAX_CTX)
    gen_ctx = min(gen_budget_kb // LLAMA_KB_PER_TOKEN, LLAMA_MAX_CTX)

    # Round down to nearest 256
    agent_ctx = (agent_ctx // 256) * 256
    gen_ctx = (gen_ctx // 256) * 256

    return max(agent_ctx, 2048), max(gen_ctx, 2048)


# ── Probe GPU once at import time ───────────────────────────────
_VRAM_MB = _get_gpu_vram_mb()
_AGENT_CTX, _GEN_CTX = _compute_context_sizes(_VRAM_MB)


@dataclass(frozen=True)
class AgentModelConfig:
    """Qwen3-4B: agentic brain for routing, JSON, tool calling."""
    MODEL_PATH: str = "models/Qwen3-4B-Q8_0.gguf"
    N_CTX: int = _AGENT_CTX
    N_GPU_LAYERS: int = -1
    TEMPERATURE: float = 0.6
    TOP_P: float = 0.95
    TOP_K: int = 20
    PRESENCE_PENALTY: float = 1.5
    MAX_TOKENS: int = 512
    REPEAT_PENALTY: float = 1.1


@dataclass(frozen=True)
class GeneratorModelConfig:
    """Llama 3.1 8B Instruct: language brain for naturalization."""
    MODEL_PATH: str = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    N_CTX: int = _GEN_CTX
    N_GPU_LAYERS: int = -1
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    MAX_TOKENS: int = 1024
    REPEAT_PENALTY: float = 1.1


@dataclass(frozen=True)
class EmbeddingConfig:
    MODEL_NAME: str = "intfloat/multilingual-e5-large-instruct"
    DIMENSION: int = 1024
    DEVICE: str = "cpu"  # CPU always — LLMs need all GPU VRAM for KV cache


@dataclass(frozen=True)
class FAQConfig:
    FAISS_INDEX_PATH: str = "faq/faq_index.faiss"
    FAQ_METADATA_PATH: str = "faq/faq_metadata.json"
    KNOWLEDGE_BASE_PATH: str = "faq/knowledge_base.json"
    HIGH_THRESHOLD: float = 0.85
    MEDIUM_THRESHOLD: float = 0.60
    TOP_K: int = 3


@dataclass(frozen=True)
class MemoryConfig:
    MAX_TURN_HISTORY: int = 16


@dataclass(frozen=True)
class AuthConfig:
    MAX_AUTH_ATTEMPTS: int = 3


@dataclass(frozen=True)
class EmotionConfig:
    ESCALATION_WINDOW: int = 3
    LEVELS: tuple = ("neutral", "concerned", "frustrated", "anxious", "angry", "stressed")


@dataclass(frozen=True)
class WorkflowConfig:
    MAX_IDLE_TURNS: int = 20
