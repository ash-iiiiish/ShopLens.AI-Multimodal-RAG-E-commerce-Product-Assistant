"""
embeddings.py — HuggingFace embeddings + FAISS index management.

Singleton pattern: model loaded once, reused across all requests.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, NORMALIZE_EMBEDDINGS,
    FAISS_INDEX_PATH, TOP_K_RESULTS
)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages HuggingFace embeddings and FAISS vector store."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(self):
        if self._ready:
            return
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
        )
        self.vectorstore: Optional[FAISS] = None
        self._ready = True
        logger.info("Embedding model ready ✅")

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def build_index(self, documents: List[Document]) -> FAISS:
        """Build FAISS index from documents and save to disk."""
        logger.info(f"Building FAISS index from {len(documents)} documents...")
        self.vectorstore = FAISS.from_documents(documents, self.model)
        self.vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info(f"Index saved to {FAISS_INDEX_PATH} ✅")
        return self.vectorstore

    def load_index(self) -> FAISS:
        """Load FAISS index from disk."""
        if not Path(FAISS_INDEX_PATH).exists():
            raise FileNotFoundError(
                f"No FAISS index at {FAISS_INDEX_PATH}. Run: python ingest.py"
            )
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            self.model,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded ✅")
        return self.vectorstore

    def get_vectorstore(self) -> FAISS:
        """Return active vectorstore, loading from disk if needed."""
        if self.vectorstore is None:
            self.load_index()
        return self.vectorstore

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        price_filter: float = None
    ) -> List[Tuple[Document, float]]:
        """
        Semantic search with optional price filter.
        Returns list of (Document, score) — lower score = more similar.
        """
        vs = self.get_vectorstore()
        results = vs.similarity_search_with_score(query, k=top_k * 3)

        if price_filter:
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("price", 0) <= price_filter
            ]

        return results[:top_k]


# Module-level singleton
embedding_manager = EmbeddingManager()
