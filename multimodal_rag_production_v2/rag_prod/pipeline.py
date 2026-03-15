"""
pipeline.py — Main RAG pipeline orchestrator.

Ties together: Vision → Retrieval → Generation.
Also provides ChatSession for multi-turn conversations.
"""
import logging
from typing import List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from vision import describe_product_image
from retriever import HybridSearcher, semantic_search, extract_price_filter
from model import generate_answer, compare_products, format_products_for_context
from config import TOP_K_RESULTS

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Structured output from a single pipeline run."""
    image_description: str
    retrieved_products: list     # List of (Document, score)
    answer: str
    price_filter_used: Optional[float] = None

    @property
    def product_names(self) -> List[str]:
        return [doc.metadata["name"] for doc, _ in self.retrieved_products]


class RAGPipeline:
    """
    Main pipeline: image + question → answer.

    Initialize with product documents for hybrid search.
    Call run() for single requests or create a ChatSession for multi-turn.
    """

    def __init__(self, documents: list = None, use_hybrid: bool = True):
        self.use_hybrid = use_hybrid and documents is not None
        if self.use_hybrid:
            logger.info("Initializing HybridSearcher (FAISS + BM25)")
            self.searcher = HybridSearcher(documents)
        else:
            logger.info("Using semantic-only search")
            self.searcher = None

    def _retrieve(self, query: str, top_k: int, price_filter: float = None) -> list:
        if self.use_hybrid:
            return self.searcher.search(query, top_k=top_k, price_filter=price_filter)
        return semantic_search(query, top_k=top_k, price_filter=price_filter)

    def run(
        self,
        image_source: Union[str, bytes, Path],
        user_question: str,
        top_k: int = TOP_K_RESULTS,
        chat_history: List[BaseMessage] = None
    ) -> PipelineResult:
        """
        Full pipeline: image → description → retrieval → answer.

        Args:
            image_source:  File path, URL, or bytes
            user_question: User's query
            top_k:         Products to retrieve
            chat_history:  Prior conversation messages

        Returns:
            PipelineResult with description, products, and answer
        """
        logger.info("=== Pipeline Start ===")

        # Step 1: Vision
        logger.info("Step 1/3: Analyzing image...")
        description = describe_product_image(image_source)

        # Step 2: Retrieval
        logger.info("Step 2/3: Retrieving products...")
        price_filter = extract_price_filter(user_question)
        combined_query = f"{description}. User wants: {user_question}"
        products = self._retrieve(combined_query, top_k=top_k, price_filter=price_filter)
        logger.info(f"Retrieved {len(products)} products")

        # Step 3: Generation
        logger.info("Step 3/3: Generating answer...")
        answer = generate_answer(
            user_question=user_question,
            image_description=description,
            retrieved_products=products,
            chat_history=chat_history
        )

        logger.info("=== Pipeline Complete ===")
        return PipelineResult(
            image_description=description,
            retrieved_products=products,
            answer=answer,
            price_filter_used=price_filter
        )

    def get_comparison(self, products: list, criteria: str = None) -> str:
        """Generate a markdown comparison table for retrieved products."""
        return compare_products(products, criteria)


class ChatSession:
    """
    Manages a single user's conversation session.
    Holds image context and chat history across multiple turns.

    Create one per user (store in Streamlit session_state).
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.chat_history: List[BaseMessage] = []
        self.current_description: Optional[str] = None
        self.current_products: Optional[list] = None

    def load_image(self, image_source: Union[str, bytes, Path]) -> str:
        """Analyze a new image and reset conversation history."""
        self.current_description = describe_product_image(image_source)
        self.current_products = self.pipeline._retrieve(
            self.current_description, top_k=TOP_K_RESULTS
        )
        self.chat_history = []
        return self.current_description

    def ask(self, question: str) -> str:
        """Ask a question about the current product. Maintains conversation history."""
        if not self.current_description:
            return "Please load a product image first."

        price_filter = extract_price_filter(question)
        combined_query = f"{self.current_description}. {question}"
        products = self.pipeline._retrieve(
            combined_query, top_k=TOP_K_RESULTS, price_filter=price_filter
        )
        self.current_products = products

        answer = generate_answer(
            user_question=question,
            image_description=self.current_description,
            retrieved_products=products,
            chat_history=self.chat_history
        )

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer

    def clear(self):
        """Reset the session."""
        self.chat_history = []
        self.current_description = None
        self.current_products = None
