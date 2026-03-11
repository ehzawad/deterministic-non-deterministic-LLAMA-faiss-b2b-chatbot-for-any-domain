"""faq -- FAQ retrieval engine (FAISS + embedding search)."""

from faq.embedder import Embedder
from faq.faq_engine import FAQEngine, FAQResult

__all__ = ["Embedder", "FAQEngine", "FAQResult"]
