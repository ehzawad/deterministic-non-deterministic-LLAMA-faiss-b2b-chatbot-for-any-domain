"""Download GGUF models from HuggingFace into the models/ directory.

Models:
  1. Qwen3-4B Q8_0  -- agentic brain (routing, JSON, tool calling)
  2. Llama 3.1 8B Instruct Q4_K_M -- generator brain (naturalization)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


# ── model manifest ─────────────────────────────────────────────────
MODELS: list[dict[str, str]] = [
    {
        "repo_id": "Qwen/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q8_0.gguf",
        "description": "Qwen3-4B Q8_0 (agentic brain)",
    },
    {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "description": "Llama 3.1 8B Instruct Q4_K_M (generator brain)",
    },
]

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_models() -> None:
    """Download every model listed in the manifest."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {MODELS_DIR}\n")

    for entry in MODELS:
        dest = MODELS_DIR / entry["filename"]

        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(
                f"[skip] {entry['description']}\n"
                f"       Already exists ({size_mb:.0f} MB): {dest}\n"
            )
            continue

        print(
            f"[download] {entry['description']}\n"
            f"           repo : {entry['repo_id']}\n"
            f"           file : {entry['filename']}"
        )

        downloaded_path = hf_hub_download(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )

        size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
        print(f"           done : {downloaded_path} ({size_mb:.0f} MB)\n")

    print("All models ready.")


if __name__ == "__main__":
    try:
        download_models()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
