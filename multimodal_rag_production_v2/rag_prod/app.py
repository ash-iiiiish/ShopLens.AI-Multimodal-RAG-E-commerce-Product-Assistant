"""
app.py — ShopLens AI · Dark Luxury Editorial UI (Responsive)
Deep obsidian base, warm gold accents, Playfair Display + DM Sans typography
Responsive: Phone / Tablet / Desktop
"""
import io
import streamlit as st
from pathlib import Path
from PIL import Image

from catalog import PRODUCT_CATALOG, build_product_text
from embeddings import embedding_manager
from pipeline import RAGPipeline, ChatSession
from langchain_core.documents import Document
from config import TOP_K_RESULTS, FAISS_INDEX_PATH

st.set_page_config(
    page_title="ShopLens AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-base:        #0b0a08;
    --bg-surface:     #131210;
    --bg-elevated:    #1a1814;
    --bg-card:        #1f1d19;
    --border-subtle:  #2a2721;
    --border-mid:     #3a3630;
    --gold:           #c9a84c;
    --gold-bright:    #e2c060;
    --gold-muted:     #8a6f2e;
    --gold-glow:      rgba(201,168,76,0.12);
    --gold-glow-sm:   rgba(201,168,76,0.07);
    --cream:          #f0ead8;
    --cream-dim:      #a89e88;
    --cream-dimmer:   #6b6254;
    --green:          #4a9a6a;
    --green-bg:       rgba(74,154,106,0.12);
    --red-accent:     #c45a3a;
    --font-display:   'Playfair Display', Georgia, serif;
    --font-body:      'DM Sans', system-ui, sans-serif;
    --radius-sm:      6px;
    --radius-md:      10px;
    --radius-lg:      16px;
    --radius-xl:      22px;
    --transition:     0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

*, *::before, *::after { box-sizing: border-box; }

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    color: var(--cream) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 70% 50% at 85% -5%,  rgba(201,168,76,0.06)  0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at -10% 95%, rgba(74,154,106,0.05)  0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50%  50%, rgba(30,28,22,0.8)     0%, transparent 70%),
        var(--bg-base) !important;
}

[data-testid="stHeader"]  { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebarContent"] { padding: 1.6rem 1.3rem !important; }

/* ── Block container ── */
.block-container {
    padding: 2.2rem 4rem 5rem !important;
    max-width: 1480px !important;
}

/* ── Noise texture overlay ── */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    opacity: 0.018;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    background-size: 180px 180px;
}

/* ═══════════════════════════════════════
   NAVBAR
═══════════════════════════════════════ */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.8rem;
    padding-bottom: 2rem;
    margin-bottom: 2.8rem;
    position: relative;
}
.navbar::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, var(--gold) 0%, var(--border-mid) 35%, transparent 100%);
}
.navbar-left  { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
.navbar-wordmark {
    font-family: var(--font-display);
    font-size: 2.2rem; font-weight: 700;
    color: var(--cream); letter-spacing: -0.03em;
    line-height: 1;
}
.navbar-wordmark span {
    color: var(--gold);
    font-style: italic;
}
.navbar-eyebrow {
    font-size: 0.65rem; font-weight: 500; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--cream-dimmer);
    position: relative; top: -1px;
}
.navbar-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 5px 14px; border-radius: 20px;
    background: var(--gold-glow);
    border: 1px solid var(--gold-muted);
    color: var(--gold);
}
.navbar-badge::before {
    content: '◆'; font-size: 0.4rem; opacity: 0.8;
}

/* ═══════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════ */
.section-header {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--cream-dimmer);
    margin-bottom: 1.1rem;
    display: flex; align-items: center; gap: 12px;
}
.section-header::before {
    content: ''; width: 20px; height: 1px;
    background: var(--gold); flex-shrink: 0;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: var(--border-subtle);
}

/* ═══════════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════════ */
[data-testid="stFileUploader"] > div {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-mid) !important;
    border-radius: var(--radius-lg) !important;
    padding: 2rem 1.5rem !important;
    transition: border-color var(--transition), background var(--transition) !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--gold-muted) !important;
    background: var(--bg-elevated) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--cream-dim) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
}

/* ═══════════════════════════════════════
   BUTTONS
═══════════════════════════════════════ */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--gold) 0%, #b8922e 100%) !important;
    color: #0b0a08 !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.7rem 1.8rem !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    transition: all var(--transition) !important;
    box-shadow:
        0 2px 16px rgba(201,168,76,0.30),
        0 1px 0 rgba(255,255,255,0.08) inset !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, var(--gold-bright) 0%, var(--gold) 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow:
        0 6px 28px rgba(201,168,76,0.42),
        0 1px 0 rgba(255,255,255,0.12) inset !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 10px rgba(201,168,76,0.20) !important;
}

/* Chip / suggestion buttons */
.chip-btn [data-testid="stButton"] > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 20px !important;
    color: var(--cream-dim) !important;
    font-size: 0.74rem !important;
    padding: 0.38rem 0.9rem !important;
    box-shadow: none !important;
    font-weight: 400 !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}
.chip-btn [data-testid="stButton"] > button:hover {
    background: var(--gold-glow) !important;
    border-color: var(--gold-muted) !important;
    color: var(--gold) !important;
    transform: none !important;
    box-shadow: 0 0 12px var(--gold-glow) !important;
}

/* ═══════════════════════════════════════
   CHAT INPUT
═══════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border-mid) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--cream) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold-muted) !important;
    box-shadow: 0 0 0 3px var(--gold-glow), 0 4px 16px rgba(0,0,0,0.4) !important;
}

/* ═══════════════════════════════════════
   CHAT MESSAGES
═══════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* ═══════════════════════════════════════
   PRODUCT CARDS
═══════════════════════════════════════ */
.product-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    margin-bottom: 10px;
    position: relative; overflow: hidden;
    transition: border-color var(--transition), box-shadow var(--transition), transform var(--transition);
}
/* Left accent bar */
.product-card::before {
    content: ''; position: absolute;
    left: 0; top: 0; bottom: 0; width: 2px;
    background: linear-gradient(180deg, var(--gold), var(--gold-muted) 60%, transparent);
}
/* Subtle top glow on hover */
.product-card::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 40%;
    background: linear-gradient(180deg, var(--gold-glow-sm), transparent);
    opacity: 0; transition: opacity var(--transition);
    pointer-events: none;
}
.product-card:hover {
    border-color: var(--gold-muted);
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), 0 0 0 1px var(--gold-glow);
    transform: translateY(-2px);
}
.product-card:hover::after { opacity: 1; }

.product-rank {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.18em;
    color: var(--gold); text-transform: uppercase; margin-bottom: 5px;
    opacity: 0.8;
}
.product-name {
    font-size: 0.9rem; font-weight: 500; color: var(--cream);
    margin-bottom: 10px; line-height: 1.4;
}
.product-meta {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.product-price {
    font-family: var(--font-display);
    font-size: 1.05rem; font-weight: 600;
    color: var(--gold);
}
.product-badge {
    font-size: 0.65rem; padding: 2px 9px; border-radius: 12px; font-weight: 500;
    letter-spacing: 0.04em;
}
.badge-match {
    background: var(--gold-glow);
    color: var(--gold); border: 1px solid var(--gold-muted);
}
.badge-cat {
    background: var(--green-bg);
    color: var(--green); border: 1px solid rgba(74,154,106,0.3);
}
.product-link {
    font-size: 0.7rem; color: var(--cream-dimmer); text-decoration: none;
    margin-top: 8px; display: inline-flex; align-items: center; gap: 4px;
    opacity: 0.6; transition: color var(--transition), opacity var(--transition);
    letter-spacing: 0.06em; text-transform: uppercase;
}
.product-link:hover { color: var(--gold); opacity: 1; }

/* ═══════════════════════════════════════
   STATUS PILLS
═══════════════════════════════════════ */
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 14px; border-radius: 20px;
    font-size: 0.74rem; font-weight: 500; letter-spacing: 0.03em;
}
.status-ready {
    background: var(--green-bg);
    border: 1px solid rgba(74,154,106,0.35);
    color: var(--green);
}
.status-waiting {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border-mid);
    color: var(--cream-dimmer);
}

/* ═══════════════════════════════════════
   INFO / EMPTY BOX
═══════════════════════════════════════ */
.info-box {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.4rem;
    font-size: 0.84rem; color: var(--cream-dim); line-height: 1.7;
}

/* ═══════════════════════════════════════
   IMAGE
═══════════════════════════════════════ */
[data-testid="stImage"] img {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-mid) !important;
    box-shadow:
        0 4px 30px rgba(0,0,0,0.6),
        0 0 0 1px rgba(201,168,76,0.08) !important;
    max-width: 100% !important;
    height: auto !important;
}

/* ═══════════════════════════════════════
   MISC
═══════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.8rem 0 !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-muted); }

/* ═══════════════════════════════════════
   MARKDOWN / TABLES
═══════════════════════════════════════ */
.stMarkdown p {
    line-height: 1.8 !important;
    color: var(--cream-dim) !important;
    font-size: 0.875rem !important;
}
.stMarkdown table {
    border-collapse: collapse !important; width: 100% !important;
    font-size: 0.82rem !important; border-radius: var(--radius-md);
    overflow: hidden;
}
.stMarkdown th {
    background: rgba(201,168,76,0.10) !important;
    color: var(--gold) !important;
    padding: 9px 14px !important;
    border: 1px solid var(--border-mid) !important;
    font-weight: 600 !important; letter-spacing: 0.05em;
}
.stMarkdown td {
    padding: 8px 14px !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--cream-dim) !important;
}
.stMarkdown tr:nth-child(even) td {
    background: rgba(255,255,255,0.02) !important;
}

/* ═══════════════════════════════════════
   EXPANDER
═══════════════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary {
    color: var(--cream-dimmer) !important;
    font-size: 0.82rem !important;
}

/* ═══════════════════════════════════════
   SIDEBAR ELEMENTS
═══════════════════════════════════════ */
[data-testid="stSlider"] label {
    color: var(--cream-dimmer) !important; font-size: 0.8rem !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--gold) !important;
    border-color: var(--gold) !important;
}
[data-testid="stCheckbox"] label {
    color: var(--cream-dimmer) !important; font-size: 0.82rem !important;
}
.sidebar-section {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--cream-dimmer);
    margin: 1.6rem 0 0.7rem;
    display: flex; align-items: center; gap: 8px;
}
.sidebar-section::after {
    content: ''; flex: 1; height: 1px; background: var(--border-subtle);
}

/* ═══════════════════════════════════════
   ALERT / SPINNER
═══════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
    font-size: 0.84rem !important;
    background: var(--bg-elevated) !important;
    border-color: var(--border-mid) !important;
}
[data-testid="stSpinner"] > div {
    border-top-color: var(--gold) !important;
}

/* ═══════════════════════════════════════
   RESPONSIVE — Tablet (≤ 1024px)
═══════════════════════════════════════ */
@media screen and (max-width: 1024px) {
    .block-container { padding: 1.6rem 1.8rem 3.5rem !important; }
    .navbar-wordmark { font-size: 1.9rem !important; }
}

/* ═══════════════════════════════════════
   RESPONSIVE — Mobile (≤ 768px)
═══════════════════════════════════════ */
@media screen and (max-width: 768px) {
    .block-container { padding: 1.1rem 1.1rem 3.5rem !important; }

    .navbar { padding-bottom: 1.3rem !important; margin-bottom: 1.8rem !important; }
    .navbar-wordmark { font-size: 1.65rem !important; }
    .navbar-eyebrow { display: none !important; }
    .navbar-badge { font-size: 0.58rem !important; padding: 4px 10px !important; }

    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important; gap: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important; flex: none !important;
    }
    [data-testid="stHorizontalBlock"] > div:first-child {
        border-bottom: 1px solid var(--border-subtle) !important;
        padding-bottom: 1.5rem !important; margin-bottom: 1.5rem !important;
    }

    .section-header { font-size: 0.57rem !important; margin-bottom: 0.8rem !important; }
    .product-card:hover { transform: none !important; }
    .product-name { font-size: 0.84rem !important; }
    [data-testid="stChatInput"] { border-radius: var(--radius-md) !important; }
    .info-box { padding: 1.4rem 1rem !important; }

    [data-testid="stHorizontalBlock"] .chip-btn [data-testid="stButton"] > button {
        font-size: 0.7rem !important; padding: 0.42rem 0.65rem !important;
    }

    .stMarkdown { overflow-x: auto !important; }
    .stMarkdown table { min-width: 380px !important; font-size: 0.76rem !important; }

    .status-pill { font-size: 0.68rem !important; padding: 4px 11px !important; }
    [data-testid="stButton"] > button {
        font-size: 0.82rem !important; padding: 0.62rem 1.2rem !important;
    }
}

/* ═══════════════════════════════════════
   RESPONSIVE — Small Phone (≤ 480px)
═══════════════════════════════════════ */
@media screen and (max-width: 480px) {
    .block-container { padding: 0.9rem 0.9rem 2.5rem !important; }
    .navbar-wordmark { font-size: 1.4rem !important; }
    .navbar-badge { display: none !important; }
    .navbar { padding-bottom: 1.1rem !important; margin-bottom: 1.4rem !important; }

    .product-card { padding: 13px 14px !important; }
    .product-name { font-size: 0.82rem !important; }
    .product-price { font-size: 0.95rem !important; }
    .product-badge { font-size: 0.6rem !important; }

    [data-testid="stFileUploader"] > div { padding: 1.3rem 1rem !important; }
    .info-box { padding: 1.2rem 0.9rem !important; font-size: 0.8rem !important; }
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Resource loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    documents = [
        Document(
            page_content=build_product_text(p),
            metadata={
                "id": p["id"], "name": p["name"], "price": p["price"],
                "brand": p["brand"], "category": p["category"], "url": p["url"]
            }
        )
        for p in PRODUCT_CATALOG
    ]
    if not Path(FAISS_INDEX_PATH).exists():
        embedding_manager.build_index(documents)
    else:
        embedding_manager.load_index()
    return RAGPipeline(documents=documents, use_hybrid=True), documents


# ── Session init ──────────────────────────────────────────────────────────────
def init_session():
    if "pipeline" not in st.session_state:
        with st.spinner("Loading models..."):
            pipeline, docs = load_pipeline()
        st.session_state.pipeline  = pipeline
        st.session_state.documents = docs
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(st.session_state.pipeline)
    if "messages"    not in st.session_state:
        st.session_state.messages = []
    if "image_ready" not in st.session_state:
        st.session_state.image_ready = False
    if "image_desc"  not in st.session_state:
        st.session_state.image_desc = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Playfair Display',serif;font-size:1.35rem;font-weight:700;
                    color:#f0ead8;margin-bottom:1.6rem;padding-bottom:1.1rem;
                    border-bottom:1px solid #2a2721;letter-spacing:-0.02em">
            Shop<span style="color:#c9a84c;font-style:italic">Lens</span> AI
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Retrieval</div>', unsafe_allow_html=True)
        top_k = st.slider("top_k", 2, 8, TOP_K_RESULTS, label_visibility="collapsed")
        st.caption(f"Returning top **{top_k}** results")

        st.markdown('<div class="sidebar-section">Debug</div>', unsafe_allow_html=True)
        show_desc = st.checkbox("Show image analysis", value=False)

        st.markdown('<div class="sidebar-section">Catalog</div>', unsafe_allow_html=True)
        cats = list({p["category"] for p in PRODUCT_CATALOG})
        st.markdown(f"""
        <div class="info-box">
            <strong style="color:#f0ead8;font-weight:600">{len(PRODUCT_CATALOG)} products</strong><br>
            <span style="font-size:0.74rem;color:#6b6254">{' · '.join(cats)}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_session.clear()
            st.session_state.messages    = []
            st.session_state.image_ready = False
            st.session_state.image_desc  = ""
            st.rerun()

        st.markdown("""
        <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #2a2721;
                    font-size:0.66rem;color:#6b6254;line-height:2.2;letter-spacing:0.04em">
            LLaMA 3.3 70B · Llama 4 Scout<br>
            HuggingFace · FAISS · LangChain
        </div>
        """, unsafe_allow_html=True)

    return top_k, show_desc


# ── Product panel ─────────────────────────────────────────────────────────────
def render_products(products: list):
    st.markdown('<div class="section-header">Similar Products Found</div>', unsafe_allow_html=True)
    for i, (doc, score) in enumerate(products, 1):
        m   = doc.metadata
        sim = int((1 / (1 + score)) * 100)
        st.markdown(f"""
        <div class="product-card">
            <div class="product-rank">Match #{i}</div>
            <div class="product-name">{m['name']}</div>
            <div class="product-meta">
                <span class="product-price">${m.get('price','N/A')}</span>
                <span class="product-badge badge-match">{sim}% similar</span>
                <span class="product-badge badge-cat">{m.get('category','')}</span>
            </div>
            <a class="product-link" href="{m.get('url','#')}" target="_blank">View product →</a>
        </div>
        """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()
    top_k, show_desc = render_sidebar()

    # Navbar
    st.markdown("""
    <div class="navbar">
        <div class="navbar-left">
            <span class="navbar-wordmark">Shop<span>Lens</span></span>
            <span class="navbar-eyebrow">E-Commerce Product Assistant</span>
        </div>
        <span class="navbar-badge">Multimodal RAG</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.65], gap="large")

    # ── LEFT ──────────────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "upload",
            type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed"
        )

        image_source = None
        if uploaded:
            image_bytes  = uploaded.read()
            image_source = image_bytes
            st.image(Image.open(io.BytesIO(image_bytes)), use_column_width=True)

        if image_source:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("✦  Analyze with Vision AI", use_container_width=True):
                with st.spinner("Analyzing image with Llama 4 Scout..."):
                    try:
                        desc = st.session_state.chat_session.load_image(image_source)
                        st.session_state.image_ready = True
                        st.session_state.image_desc  = desc
                        st.session_state.messages    = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Vision error: {e}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.session_state.image_ready:
            st.markdown('<div class="status-pill status-ready">✓ Ready — ask your questions</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill status-waiting">○ Upload an image to begin</div>', unsafe_allow_html=True)

        if show_desc and st.session_state.image_desc:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            with st.expander("Image analysis"):
                st.markdown(f"""
                <div style="font-size:0.82rem;color:#a89e88;line-height:1.8;font-family:'DM Sans',sans-serif">
                    {st.session_state.image_desc}
                </div>""", unsafe_allow_html=True)

        if st.session_state.image_ready and st.session_state.chat_session.current_products:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            render_products(st.session_state.chat_session.current_products[:top_k])

    # ── RIGHT ─────────────────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)

        if not st.session_state.image_ready:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:4.5rem 2.5rem;background:#131210;
                         border-color:#2a2721;position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;
                             background:linear-gradient(90deg,transparent,#c9a84c,transparent);
                             opacity:0.5"></div>
                <div style="font-size:2.8rem;margin-bottom:1.3rem;line-height:1;filter:drop-shadow(0 0 18px rgba(201,168,76,0.4))">
                    🛍️
                </div>
                <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:600;
                            color:#f0ead8;margin-bottom:0.7rem;letter-spacing:-0.02em;line-height:1.2">
                    Upload a product image
                </div>
                <div style="font-size:0.84rem;color:#6b6254;line-height:1.85;
                            max-width:300px;margin:0 auto">
                    Drop any product photo — sneakers, laptops, phones, clothing.
                    Ask about specs, find alternatives, and compare prices.
                </div>
                <div style="margin-top:2rem;display:flex;justify-content:center;gap:10px;flex-wrap:wrap">
                    <span style="font-size:0.68rem;padding:4px 12px;border-radius:12px;
                                 background:rgba(201,168,76,0.10);color:#c9a84c;
                                 border:1px solid rgba(201,168,76,0.25);letter-spacing:0.04em">
                        Find similar products
                    </span>
                    <span style="font-size:0.68rem;padding:4px 12px;border-radius:12px;
                                 background:rgba(74,154,106,0.10);color:#4a9a6a;
                                 border:1px solid rgba(74,154,106,0.25);letter-spacing:0.04em">
                        Compare prices
                    </span>
                    <span style="font-size:0.68rem;padding:4px 12px;border-radius:12px;
                                 background:rgba(255,255,255,0.04);color:#a89e88;
                                 border:1px solid #2a2721;letter-spacing:0.04em">
                        Get specs
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Suggestion chips — 2 cols on mobile, 4 on desktop
            suggestions = ["What are the features?", "Under $300?", "Compare top 2", "Best value?"]
            chip_cols = st.columns(4)
            for i, (col, sug) in enumerate(zip(chip_cols, suggestions)):
                with col:
                    st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                    if st.button(sug, key=f"chip_{i}", use_container_width=True):
                        st.session_state._pending = sug
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center;padding:2.5rem 0;color:#3a3630;font-size:0.82rem;
                             letter-spacing:0.04em">
                    No messages yet — try a suggestion or type your question
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            pending    = st.session_state.pop("_pending", None)
            user_input = st.chat_input("Ask about this product...") or pending

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner(""):
                        try:
                            is_compare = any(
                                w in user_input.lower()
                                for w in ["compare", "versus", "vs", "difference"]
                            )
                            if is_compare:
                                reply = st.session_state.pipeline.get_comparison(
                                    st.session_state.chat_session.current_products,
                                    criteria=user_input
                                )
                            else:
                                reply = st.session_state.chat_session.ask(user_input)
                            st.markdown(reply)
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.error(f"Error: {e}")
                st.rerun()


if __name__ == "__main__":
    main()
    