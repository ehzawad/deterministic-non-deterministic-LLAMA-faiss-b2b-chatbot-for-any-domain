"""DualLLMEngine -- manages both Qwen3-4B (agent) and Llama 3.1 8B (generator).

GPU-aware: automatically retries with halved context if a model fails to load
(e.g., VRAM too tight). Works on T4 (15GB), L4 (24GB), L40 (48GB), A100 (80GB).
"""

from __future__ import annotations

import logging
from typing import Sequence

from llama_cpp import Llama

from config import AgentModelConfig, GeneratorModelConfig

log = logging.getLogger(__name__)


def _load_model(path: str, n_ctx: int, n_gpu_layers: int, label: str) -> tuple[Llama, int]:
    """Load a GGUF model, retrying with smaller context on failure."""
    ctx = n_ctx
    while ctx >= 2048:
        try:
            print(f"  Loading {label} (n_ctx={ctx})...")
            model = Llama(
                model_path=path,
                n_ctx=ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            print(f"  {label} loaded (n_ctx={ctx})")
            return model, ctx
        except (ValueError, RuntimeError) as e:
            log.warning("%s failed with n_ctx=%d: %s. Halving context.", label, ctx, e)
            ctx //= 2
    raise RuntimeError(f"Cannot load {label} even with n_ctx=2048")


class DualLLMEngine:
    """Thin wrapper around two llama-cpp Llama instances."""

    def __init__(
        self,
        agent_config: AgentModelConfig,
        generator_config: GeneratorModelConfig,
    ) -> None:
        self._agent_cfg = agent_config
        self._gen_cfg = generator_config

        self._agent, self.agent_ctx = _load_model(
            agent_config.MODEL_PATH,
            agent_config.N_CTX,
            agent_config.N_GPU_LAYERS,
            "Qwen3-4B (agent)",
        )

        self._generator, self.generator_ctx = _load_model(
            generator_config.MODEL_PATH,
            generator_config.N_CTX,
            generator_config.N_GPU_LAYERS,
            "Llama 3.1 8B (generator)",
        )

    def _generate(
        self,
        model: Llama,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        stop: Sequence[str] | None,
    ) -> str:
        result = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=list(stop) if stop else None,
        )
        text: str = result["choices"][0]["text"]
        return text

    def agent_generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.6,
    ) -> str:
        """Generate with Qwen3 using chat completion (proper chat format)."""
        result = self._agent.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self._agent_cfg.TOP_P,
            repeat_penalty=self._agent_cfg.REPEAT_PENALTY,
        )
        text: str = result["choices"][0]["message"]["content"]
        return text

    def generator_generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Sequence[str] | None = None,
    ) -> str:
        return self._generate(
            self._generator, prompt,
            max_tokens=max_tokens, temperature=temperature,
            top_p=self._gen_cfg.TOP_P,
            repeat_penalty=self._gen_cfg.REPEAT_PENALTY,
            stop=stop,
        )

    def agent_count_tokens(self, text: str) -> int:
        return len(self._agent.tokenize(text.encode("utf-8"), add_bos=False))

    def generator_count_tokens(self, text: str) -> int:
        return len(self._generator.tokenize(text.encode("utf-8"), add_bos=False))
