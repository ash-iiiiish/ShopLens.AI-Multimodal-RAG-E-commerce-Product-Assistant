"""
ingest.py — Data ingestion pipeline.

Run ONCE (or whenever the catalog changes) to build the FAISS index.

Usage:
    python ingest.py              # uses default PRODUCT_CATALOG
    python ingest.py --source path/to/catalog.json
"""
import json
import logging
import argparse

from langchain_core.documents import Document

from catalog import PRODUCT_CATALOG, build_product_text, save_catalog
from embeddings import embedding_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_documents(products: list) -> list:
    """Convert product dicts into LangChain Documents."""
    docs = []
    for p in products:
        docs.append(Document(
            page_content=build_product_text(p),
            metadata={
                "id":       p["id"],
                "name":     p["name"],
                "price":    p.get("price", 0),
                "brand":    p.get("brand", ""),
                "category": p.get("category", ""),
                "url":      p.get("url", ""),
            }
        ))
    return docs


def ingest(products: list) -> int:
    """Build FAISS index and save catalog. Returns count of indexed products."""
    logger.info(f"Starting ingestion of {len(products)} products...")

    documents = build_documents(products)
    embedding_manager.build_index(documents)
    save_catalog(products)

    logger.info(f"✅ Done: {len(documents)} products indexed.")
    return len(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest products into FAISS index")
    parser.add_argument("--source", default="default",
                        help="'default' or path to a JSON catalog file")
    args = parser.parse_args()

    if args.source == "default":
        products = PRODUCT_CATALOG
    else:
        with open(args.source) as f:
            products = json.load(f)

    count = ingest(products)
    print(f"\n✅ {count} products indexed and ready.")
