<div align="center">

# 🔍 ShopLens AI
### Multimodal RAG for E-Commerce Product Assistant::::

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat-square)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=flat-square)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<p align="center">
  <strong>Upload any product image → Get specs, find alternatives, compare prices</strong><br>
  Powered by vision LLMs, semantic search, and conversational AI — all in one app.
</p>

---

</div>

## 📌 Overview

**ShopLens AI** is a production-ready multimodal RAG (Retrieval-Augmented Generation) system that combines computer vision, vector search, and large language models to build an intelligent e-commerce shopping assistant.

A user uploads a product photo — a pair of sneakers, a laptop, headphones — and the system:
1. **Understands the image** using Llama 4 Scout vision model
2. **Finds similar products** using hybrid semantic + keyword search over a FAISS vector database
3. **Answers questions** in natural language using ChatGroq (LLaMA 3.3 70B)
4. **Remembers context** across a multi-turn conversation

> Think of it as **ChatGPT + Google Lens + Amazon Search** combined into one interface.

---

![Demo](images/image.png)


---

## 🏗️ Architecture

```
User uploads product image
         │
         ▼
┌─────────────────────┐
│     vision.py       │  Llama 4 Scout (Groq) → Rich product description
│  Image → Text       │  Caches results by image hash
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    retriever.py     │  FAISS semantic search
│  Hybrid Search      │  + BM25 keyword search
│  (RRF Fusion)       │  → Reciprocal Rank Fusion
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      model.py       │  ChatGroq LLaMA 3.3 70B
│  RAG Generation     │  + Conversation memory
│                     │  → Natural language answer
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      app.py         │  Streamlit dark UI
│  Web Interface      │  Multi-turn chat + product cards
└─────────────────────┘
```

---

## 🚀 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vision LLM** | Llama 4 Scout 17B (Groq) | Image → product description |
| **Chat LLM** | LLaMA 3.3 70B (Groq) | Answer generation |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Text → vectors |
| **Vector DB** | FAISS (CPU) | Semantic similarity search |
| **Keyword Search** | BM25 (rank-bm25) | Exact brand/model matching |
| **Fusion** | Reciprocal Rank Fusion | Hybrid retrieval ranking |
| **Framework** | LangChain 0.3 | LLM orchestration |
| **UI** | Streamlit | Web interface |

---

## 📁 Project Structure

```
rag_prod/
│
├── app.py           → Streamlit web UI (dark premium theme)
├── pipeline.py      → Main orchestrator + ChatSession (multi-turn)
│
├── vision.py        → Image loading, format conversion, Groq vision call
├── retriever.py     → Hybrid FAISS + BM25 search with RRF fusion
├── embeddings.py    → HuggingFace model + FAISS index build/load/search
├── model.py         → ChatGroq LLM — answer generation + comparison table
│
├── catalog.py       → Product data + searchable text builder
├── ingest.py        → One-time index builder CLI
├── config.py        → All settings loaded from .env
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/shoplens-ai.git
cd shoplens-ai/rag_prod
```

### 2. Create a virtual environment

```bash
python -m venv freshenv

# Windows
freshenv\Scripts\activate

# Mac / Linux
source freshenv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **NumPy note:** If you get a NumPy 2.x compatibility error with FAISS, run:
> ```bash
> pip install "numpy<2" && pip install faiss-cpu --force-reinstall
> ```

### 4. Set up API key

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys) — no credit card required.

### 5. Build the vector index

```bash
python ingest.py
```

Expected output:
```
INFO - Building FAISS index from 8 documents...
INFO - Index saved to faiss_product_index ✅
✅ Done: 8 products indexed and ready.
```

### 6. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎯 Features

### Core Pipeline
- **Multimodal input** — Upload product images (JPG, PNG, WEBP, AVIF auto-converted)
- **Vision understanding** — Llama 4 Scout extracts category, brand, color, features, use case
- **Hybrid retrieval** — FAISS semantic search + BM25 keyword search fused with RRF
- **Conversational AI** — Multi-turn chat with windowed conversation memory
- **Price filtering** — "under $300" auto-detected from natural language

### Advanced Retrieval
- **Reciprocal Rank Fusion (RRF)** — Merges semantic and keyword rankings without score normalization
- **Image description caching** — MD5 hash-based cache avoids redundant Vision API calls
- **Format normalization** — AVIF, BMP, TIFF automatically converted to JPEG before processing
- **Post-retrieval filtering** — Metadata filters applied after embedding search

### UI / UX
- **Dark premium theme** — `#0a0a0f` background with purple/orange accent gradients
- **Product cards** — Match score, price, category, and direct product links
- **Suggestion chips** — One-click common queries
- **Comparison mode** — Auto-detects "compare" intent and generates markdown tables
- **Session management** — Clear conversation without page reload

---

## 💡 Key Concepts

### Why Hybrid Search?

| Search Type | Strength | Weakness |
|------------|----------|----------|
| Semantic (FAISS) | Understands meaning — "running shoes" matches "athletic footwear" | Misses exact brand names |
| Keyword (BM25) | Finds exact matches — "Nike Air Max 270" | No semantic understanding |
| **Hybrid (RRF)** | **Best of both worlds** | — |

### Reciprocal Rank Fusion
RRF combines two ranked lists without requiring score normalization:
```
RRF_score(doc) = Σ 1 / (k + rank_i)
```
where `k=60` is the standard constant from the original paper.

### RAG Pipeline
```
Query = image_description + user_question
   ↓
Retrieve top-K products from vector store
   ↓
Format as context: [Product name, specs, price, link]
   ↓
LLM generates answer grounded in retrieved context
```

---

## 🔧 Configuration

All settings are in `config.py`, overridable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Required. Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Chat LLM |
| `VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision LLM |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `TOP_K_RESULTS` | `5` | Products to retrieve |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `LLM_TEMPERATURE` | `0.3` | Lower = more factual |

---

## 📦 Adding More Products

**From a JSON file:**
```bash
python ingest.py --source path/to/catalog.json
```

JSON format:
```json
[
  {
    "id": "P001",
    "name": "Product Name",
    "category": "Category",
    "brand": "Brand",
    "price": 199,
    "specs": "Technical specifications...",
    "description": "Product description...",
    "url": "https://example.com/product"
  }
]
```

**Upgrade embedding model for better accuracy:**
```python
# In config.py
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-dim, higher accuracy
```

---

## 🛠️ Troubleshooting

| Error | Fix |
|-------|-----|
| `GROQ_API_KEY not found` | Check `.env` file is in the `rag_prod/` folder |
| `No FAISS index found` | Run `python ingest.py` before starting the app |
| `numpy.core.multiarray failed` | Run `pip install "numpy<2"` |
| `fbgemm.dll not found` | Install [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `proxies` TypeError (groq) | Run `pip install groq==0.9.0 httpx==0.27.0` |
| Vision AVIF error | Run `pip install pillow-avif-plugin` |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---


## 👨‍💻 Contributors
- [@ash-iiiiish](https://github.com/ash-iiiiish)


## 🤝 Contributing
Contributions are welcome! Fork this repository and submit a pull request.


---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.............

---




