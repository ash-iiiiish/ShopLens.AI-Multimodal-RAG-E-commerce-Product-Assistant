"""
model.py — ChatGroq LLM for answer generation and product comparison.
"""
import logging
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

from config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, MAX_CHAT_HISTORY

logger = logging.getLogger(__name__)

# Module-level LLM instance (reused across all calls)
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
    groq_api_key=GROQ_API_KEY
)

SYSTEM_PROMPT = """You are a helpful AI shopping assistant for an e-commerce platform.
You help users find products based on images they upload and questions they ask.
Always be specific, helpful, and honest. If you don't know something, say so.
Format your response clearly with product names, prices, and key differences."""


def format_products_for_context(results_with_scores: list) -> str:
    """Convert retrieved (Document, score) pairs into readable LLM context string."""
    if not results_with_scores:
        return "No matching products found."

    parts = []
    for i, (doc, score) in enumerate(results_with_scores, 1):
        m = doc.metadata
        similarity = 1 / (1 + score)
        parts.append(
            f"Product {i}: {m['name']}\n"
            f"  Brand: {m.get('brand','N/A')} | Category: {m.get('category','N/A')} "
            f"| Price: ${m.get('price','N/A')} | Match: {similarity:.0%}\n"
            f"  Details: {doc.page_content}\n"
            f"  Link: {m.get('url','N/A')}"
        )
    return "\n\n".join(parts)


def generate_answer(
    user_question: str,
    image_description: str,
    retrieved_products: list,
    chat_history: List[BaseMessage] = None
) -> str:
    """
    RAG generation step: context + question → answer.

    Args:
        user_question:      What the user asked
        image_description:  Text from vision module
        retrieved_products: List of (Document, score) from retriever
        chat_history:       Prior messages for multi-turn support
    """
    product_context = format_products_for_context(retrieved_products)

    user_prompt = f"""The user uploaded a product image. Here's what the image shows:

IMAGE DESCRIPTION:
{image_description}

SIMILAR PRODUCTS FOUND IN CATALOG:
{product_context}

USER QUESTION:
{user_question}

Please answer the question using the product information above.
If alternatives are requested, list them with prices.
If specs are requested, use the product details provided.
End with a helpful buying recommendation."""

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    if chat_history:
        messages.extend(chat_history[-MAX_CHAT_HISTORY:])

    messages.append(HumanMessage(content=user_prompt))

    response = llm.invoke(messages)
    return response.content


def compare_products(product_docs: list, user_criteria: str = None) -> str:
    """
    Generate a markdown comparison table for top products.

    Args:
        product_docs:  List of (Document, score) pairs
        user_criteria: Optional focus (e.g. 'value for money')
    """
    products_text = "\n\n".join([
        f"Product {i+1}: {doc.metadata['name']}\n"
        f"Price: ${doc.metadata.get('price', 'N/A')}\n"
        f"Details: {doc.page_content}"
        for i, (doc, _) in enumerate(product_docs[:3])
    ])

    criteria_text = f"Focus on: {user_criteria}" if user_criteria else "Compare all key aspects."

    prompt = f"""Compare these products as a shopping assistant.
{criteria_text}

{products_text}

Create a markdown comparison table with rows for: Price, Key Features, Best For, Pros, Cons.
Then give a 2-sentence recommendation."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
