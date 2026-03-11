"""Three-tier FAQ retrieval engine backed by FAISS."""

from __future__ import annotations

import json
from dataclasses import dataclass

import faiss
import numpy as np

from config import FAQConfig
from faq.embedder import Embedder


@dataclass
class FAQResult:
    """Outcome of a single FAQ lookup.

    Attributes:
        tier: ``"PURE_FAQ"`` when the best match exceeds the high threshold,
              ``"BLENDED"`` when it falls between the medium and high
              thresholds, or ``"PURE_LLM"`` when nothing matches well.
        confidence: cosine similarity of the best FAISS hit (0-1).
        question: matched FAQ question (``None`` for PURE_LLM).
        answer: matched FAQ answer (``None`` for PURE_LLM).
        category: FAQ category of the match (``None`` for PURE_LLM).
        hedged: ``True`` when the tier is ``BLENDED`` -- the caller should
                treat the answer as a strong hint rather than verbatim truth.
    """

    tier: str          # "PURE_FAQ" | "BLENDED" | "PURE_LLM"
    confidence: float
    question: str | None
    answer: str | None
    category: str | None
    hedged: bool


class FAQEngine:
    """FAISS-backed FAQ search with three confidence tiers.

    Tier logic (based on cosine similarity of the best match):
        >  ``config.HIGH_THRESHOLD``   -> **PURE_FAQ**   (return as-is)
        >= ``config.MEDIUM_THRESHOLD`` -> **BLENDED**    (hedged answer)
        <  ``config.MEDIUM_THRESHOLD`` -> **PURE_LLM**   (no FAQ match)
    """

    def __init__(self, embedder: Embedder, config: FAQConfig | None = None) -> None:
        self.config = config or FAQConfig()
        self.embedder = embedder

        # Load FAISS index
        self.index: faiss.IndexFlatIP = faiss.read_index(self.config.FAISS_INDEX_PATH)

        # Load metadata (same row order as the index)
        with open(self.config.FAQ_METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata: list[dict] = json.load(f)

        assert self.index.ntotal == len(self.metadata), (
            f"Index/metadata mismatch: {self.index.ntotal} vectors vs "
            f"{len(self.metadata)} metadata entries"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, resolved_query: str) -> FAQResult:
        """Search the FAQ index and classify the result into a tier.

        Parameters:
            resolved_query: the user query after any coreference /
                pronoun resolution performed upstream.

        Returns:
            An :class:`FAQResult` describing the best match and its tier.
        """
        # 1. Embed the query
        query_vec = self.embedder.embed_query(resolved_query)
        query_vec = np.expand_dims(query_vec, axis=0)       # (1, dim)

        # 2. Search top-K
        similarities, indices = self.index.search(query_vec, self.config.TOP_K)

        best_sim: float = float(similarities[0][0])
        best_idx: int = int(indices[0][0])

        # Guard against empty index edge-case
        if best_idx < 0:
            return FAQResult(
                tier="PURE_LLM",
                confidence=0.0,
                question=None,
                answer=None,
                category=None,
                hedged=False,
            )

        match = self.metadata[best_idx]

        # 3. Tier classification
        if best_sim > self.config.HIGH_THRESHOLD:
            return FAQResult(
                tier="PURE_FAQ",
                confidence=best_sim,
                question=match["question"],
                answer=match["answer"],
                category=match["category"],
                hedged=False,
            )

        if best_sim >= self.config.MEDIUM_THRESHOLD:
            return FAQResult(
                tier="BLENDED",
                confidence=best_sim,
                question=match["question"],
                answer=match["answer"],
                category=match["category"],
                hedged=True,
            )

        return FAQResult(
            tier="PURE_LLM",
            confidence=best_sim,
            question=None,
            answer=None,
            category=None,
            hedged=False,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def search_top_k(
        self, resolved_query: str, k: int | None = None,
    ) -> list[tuple[float, dict]]:
        """Return the top-K matches as ``(similarity, metadata)`` pairs.

        Useful for debugging or for the generator model to see multiple
        candidate answers.
        """
        k = k or self.config.TOP_K
        query_vec = self.embedder.embed_query(resolved_query)
        query_vec = np.expand_dims(query_vec, axis=0)

        similarities, indices = self.index.search(query_vec, k)

        results: list[tuple[float, dict]] = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
            results.append((float(sim), self.metadata[int(idx)]))
        return results
