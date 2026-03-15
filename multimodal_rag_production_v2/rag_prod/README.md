# 🛍️ Multimodal RAG E-Commerce Assistant : <br>
Upload any product image → get specs, find alternatives, compare prices. All in natural language.

**Stack:** ChatGroq (LLaMA 3.3 70B) · Llama 4 Scout (Vision) · HuggingFace Embeddings · FAISS · LangChain · Streamlit

[Demo](images\image.png)

## Project Structure

```
├── config.py        — All settings, loaded from .env
├── catalog.py       — Product data + text builder
├── embeddings.py    — HuggingFace model + FAISS index manager
├── vision.py        — Image → text (Llama 4 Scout on Groq)
├── retriever.py     — Hybrid FAISS + BM25 with RRF
├── model.py         — ChatGroq generation + comparison
├── pipeline.py      — Orchestrator + ChatSession
├── ingest.py        — Build FAISS index from catalog
├── app.py           — Streamlit UI
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add your key
cp .env.example .env
# Edit .env → GROQ_API_KEY=gsk_yourkey

# 3. Build the vector index
python ingest.py

# 4. Run
streamlit run app.py
```

## Pipeline Flow

```
Image upload
    → vision.py       (Llama 4 Scout → product description)
    → retriever.py    (FAISS + BM25 + RRF → top-K products)
    → model.py        (ChatGroq → final answer)
    → app.py          (Streamlit → display)
```
