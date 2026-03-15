"""
retriever.py — Retrieval logic: semantic (FAISS) + keyword (BM25) hybrid search.

Uses Reciprocal Rank Fusion (RRF) to merge both ranking lists.
Also provides price filter extraction from natural language.
"""
import re
import json
import logging
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from embeddings import embedding_manager
from config import TOP_K_RESULTS, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


# ── Price extraction ──────────────────────────────────────────────────────────

_PRICE_PATTERNS = [
    r'under\s*\$?([\d,]+)',
    r'below\s*\$?([\d,]+)',
    r'less\s+than\s*\$?([\d,]+)',
    r'max(?:imum)?\s*\$?([\d,]+)',
    r'budget\s*(?:of\s*)?\$?([\d,]+)',
    r'\$?([\d,]+)\s*or\s*less',
    r'within\s*\$?([\d,]+)',
]


def extract_price_filter(text: str) -> Optional[float]:
    """Extract a price ceiling from natural language text."""
    for pattern in _PRICE_PATTERNS:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1).replace(',', ''))
    return None


# ── Semantic-only retrieval ───────────────────────────────────────────────────

def semantic_search(
    query: str,
    top_k: int = TOP_K_RESULTS,
    price_filter: float = None
) -> List[Tuple[Document, float]]:
    """Pure FAISS semantic search with optional price filter."""
    return embedding_manager.similarity_search(
        query=query,
        top_k=top_k,
        price_filter=price_filter
    )


# ── Hybrid retrieval (FAISS + BM25 + RRF) ────────────────────────────────────

class HybridSearcher:
    """
    Combines FAISS semantic search + BM25 keyword search via Reciprocal Rank Fusion.

    Why hybrid?
    - Semantic: 'running shoes' matches 'athletic footwear' (meaning-aware)
    - BM25: 'Nike Air Max 270' finds exact brand/model matches
    - RRF: merges rankings without score normalization

    k=60 is the standard RRF constant (from the original paper).
    """

    def __init__(self, documents: List[Document], rrf_k: int = 60):
        self.documents = documents
        self.rrf_k = rrf_k
        self._bm25 = None
        self._build_bm25()

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.page_content.lower().split() for doc in self.documents]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built ✅")
        except ImportError:
            logger.warning("rank-bm25 not installed — falling back to semantic-only search.")

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        price_filter: float = None
    ) -> List[Tuple[Document, float]]:
        """Hybrid search with RRF. Falls back to semantic if BM25 unavailable."""
        if self._bm25 is None:
            return semantic_search(query, top_k, price_filter)

        fetch_k = top_k * 4

        # Semantic results
        sem_results = embedding_manager.similarity_search(query, top_k=fetch_k)
        sem_ids = [doc.metadata['id'] for doc, _ in sem_results]

        # BM25 results
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranked = np.argsort(bm25_scores)[::-1][:fetch_k]
        bm25_ids = [self.documents[i].metadata['id'] for i in bm25_ranked]

        # Reciprocal Rank Fusion
        rrf: dict = {}
        for rank, doc_id in enumerate(sem_ids):
            rrf[doc_id] = rrf.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)
        for rank, doc_id in enumerate(bm25_ids):
            rrf[doc_id] = rrf.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)

        top_ids = sorted(rrf, key=rrf.get, reverse=True)

        id_to_doc = {doc.metadata['id']: doc for doc in self.documents}
        results = []
        for doc_id in top_ids:
            if doc_id not in id_to_doc:
                continue
            doc = id_to_doc[doc_id]
            if price_filter and doc.metadata.get('price', 0) > price_filter:
                continue
            results.append((doc, 1 - rrf[doc_id]))  # invert so lower = better
            if len(results) >= top_k:
                break

        return results


# ── Intent extraction ─────────────────────────────────────────────────────────

def extract_search_intent(user_question: str) -> dict:
    """
    Use ChatGroq to extract structured search intent from natural language.
    Returns dict with: category, max_price, min_price, brands, color, use_case, intent
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=256,
        groq_api_key=GROQ_API_KEY
    )

    prompt = f"""Extract search parameters from this shopping query. Return ONLY valid JSON.

Query: "{user_question}"

JSON:
{{
  "category": "product type or null",
  "max_price": number_or_null,
  "min_price": number_or_null,
  "brands": ["list"] or null,
  "color": "color or null",
  "use_case": "use case or null",
  "intent": "find_similar|get_specs|compare|recommend"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = re.sub(r'^```(?:json)?\s*', '', response.content.strip())
        text = re.sub(r'\s*```$', '', text).strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Intent extraction failed: {e}")
        return {"intent": "find_similar"}
