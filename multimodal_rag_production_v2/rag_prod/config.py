"""
config.py — Central configuration
All settings loaded from .env — never hardcode keys here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
DATA_DIR         = BASE_DIR / "data"
FAISS_INDEX_DIR  = BASE_DIR / "faiss_product_index"
UPLOAD_DIR       = BASE_DIR / "uploads"

for d in [DATA_DIR, FAISS_INDEX_DIR, UPLOAD_DIR]:
    d.mkdir(exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found! Add it to your .env file.")

# ── Models ────────────────────────────────────────────────────────────────────
GROQ_MODEL      = "llama-3.3-70b-versatile"
VISION_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS        = 5
EMBEDDING_DEVICE     = "cpu"          # change to "cuda" if GPU available
NORMALIZE_EMBEDDINGS = True
MAX_CHAT_HISTORY     = 6              # last N messages kept in context window

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_TEMPERATURE  = 0.3
LLM_MAX_TOKENS   = 1024
VISION_MAX_TOKENS = 512
VISION_TEMPERATURE = 0.2

# ── Paths ─────────────────────────────────────────────────────────────────────
CATALOG_PATH     = DATA_DIR / "product_catalog.json"
FAISS_INDEX_PATH = str(FAISS_INDEX_DIR)
