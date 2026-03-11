"""E5 multilingual embedding wrapper for FAQ retrieval."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


_QUERY_INSTRUCTION = (
    "Instruct: Retrieve the most relevant banking FAQ answer\nQuery: "
)
_PASSAGE_PREFIX = "passage: "


class Embedder:
    """Thin wrapper around intfloat/multilingual-e5-large-instruct.

    E5-instruct models expect a task-specific instruction prepended to
    *queries* and a plain ``passage:`` prefix for *documents*.  Normalizing
    all vectors lets us use inner-product search as cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        device: str = "cuda",
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension: int = self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query.

        Returns a normalized 1-D vector of shape ``(dimension,)``.
        The E5-instruct format prepends a task instruction so the model
        knows this is a *search* query rather than a passage.
        """
        formatted = f"{_QUERY_INSTRUCTION}{query}"
        vector: np.ndarray = self.model.encode(
            formatted, normalize_embeddings=True, show_progress_bar=False,
        )
        return vector.astype(np.float32)

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Embed a batch of FAQ passages / questions.

        Returns a normalized matrix of shape ``(N, dimension)`` where *N* is
        ``len(documents)``.  Each text is prefixed with ``passage: `` per
        the E5 convention for indexable passages.
        """
        formatted = [f"{_PASSAGE_PREFIX}{doc}" for doc in documents]
        vectors: np.ndarray = self.model.encode(
            formatted, normalize_embeddings=True, show_progress_bar=True,
        )
        return vectors.astype(np.float32)
