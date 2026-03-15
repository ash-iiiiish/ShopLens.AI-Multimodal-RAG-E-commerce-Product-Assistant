"""
app.py — ShopLens AI · Light Mode UI (Responsive)
Warm editorial design: cream base, terracotta accents, refined typography
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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #faf8f4 !important;
    color: #1a1814 !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 60% 40% at 90% 0%, rgba(210,140,90,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 0% 100%, rgba(180,210,190,0.10) 0%, transparent 55%),
        #faf8f4 !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] {
    background: #f4f1eb !important;
    border-right: 1px solid #e8e2d6 !important;
}
section[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Desktop block container ── */
.block-container {
    padding: 2rem 3.5rem 4rem !important;
    max-width: 1400px !important;
}

/* ── Navbar ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.6rem;
    padding-bottom: 1.8rem;
    border-bottom: 1.5px solid #e8e2d6;
    margin-bottom: 2.5rem;
}
.navbar-left  { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.navbar-wordmark {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem; font-weight: 600;
    color: #1a1814; letter-spacing: -0.02em;
}
.navbar-wordmark span { color: #c4622d; }
.navbar-sub {
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: #9b8e7e;
}
.navbar-pill {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 12px; border-radius: 20px;
    background: #fdf0e8; border: 1px solid #e8c4a0; color: #c4622d;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] > div {
    background: #fff !important;
    border: 1.5px dashed #d4c4b0 !important;
    border-radius: 14px !important; padding: 1.8rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #c4622d !important;
    background: #fdf8f4 !important;
}
[data-testid="stFileUploader"] label {
    color: #7a6e62 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: #c4622d !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 0.65rem 1.6rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important; font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(196,98,45,0.25) !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    background: #b0541f !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(196,98,45,0.35) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* Chip buttons */
.chip-btn [data-testid="stButton"] > button {
    background: #fff !important;
    border: 1px solid #e0d8ce !important;
    border-radius: 20px !important; color: #5a5048 !important;
    font-size: 0.76rem !important; padding: 0.35rem 0.85rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    font-weight: 400 !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.chip-btn [data-testid="stButton"] > button:hover {
    background: #fdf0e8 !important;
    border-color: #c4622d !important; color: #c4622d !important;
    transform: none !important;
    box-shadow: 0 1px 6px rgba(196,98,45,0.15) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #fff !important;
    border: 1.5px solid #e0d8ce !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #1a1814 !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #c4622d !important;
    box-shadow: 0 0 0 3px rgba(196,98,45,0.08), 0 2px 8px rgba(0,0,0,0.04) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important; padding: 0.3rem 0 !important;
}

/* ── Product cards ── */
.product-card {
    background: #fff;
    border: 1px solid #ede8e0;
    border-radius: 14px; padding: 15px 17px; margin-bottom: 10px;
    transition: all 0.2s ease; position: relative; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.product-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, #c4622d, #e8a878);
    border-radius: 3px 0 0 3px;
}
.product-card:hover {
    border-color: #d4a08a;
    box-shadow: 0 4px 16px rgba(196,98,45,0.12);
    transform: translateY(-1px);
}
.product-rank {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em;
    color: #c4622d; text-transform: uppercase; margin-bottom: 4px;
}
.product-name {
    font-size: 0.88rem; font-weight: 500; color: #1a1814;
    margin-bottom: 8px; line-height: 1.35;
}
.product-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.product-price { font-size: 1.05rem; font-weight: 600; color: #2a7a4a; }
.product-badge { font-size: 0.68rem; padding: 2px 8px; border-radius: 12px; font-weight: 500; }
.badge-match { background: #fdf0e8; color: #c4622d; border: 1px solid #edc4a4; }
.badge-cat   { background: #f0f4f0; color: #4a6a50; border: 1px solid #c4d8c4; }
.product-link {
    font-size: 0.72rem; color: #c4622d; text-decoration: none;
    margin-top: 7px; display: inline-flex; align-items: center; gap: 3px;
    opacity: 0.7; transition: opacity 0.15s;
}
.product-link:hover { opacity: 1; }

/* ── Section headers ── */
.section-header {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #b0a090; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 10px;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: #ede8e0; }

/* ── Status pills ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 13px; border-radius: 20px; font-size: 0.76rem; font-weight: 500;
}
.status-ready   { background: #edf7f1; border: 1px solid #b8dfc8; color: #2a7a4a; }
.status-waiting { background: #f4f1eb; border: 1px solid #e0d8ce; color: #9b8e7e; }

/* ── Info / empty box ── */
.info-box {
    background: #fff; border: 1px solid #ede8e0;
    border-radius: 14px; padding: 1.1rem 1.3rem;
    font-size: 0.84rem; color: #7a6e62; line-height: 1.65;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid #ede8e0 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    max-width: 100% !important;
    height: auto !important;
}

/* ── Misc ── */
hr { border-color: #ede8e0 !important; margin: 1.5rem 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #d4c4b0; border-radius: 2px; }

/* ── Markdown tables ── */
.stMarkdown p { line-height: 1.75 !important; color: #2a2420 !important; font-size: 0.88rem !important; }
.stMarkdown table { border-collapse: collapse !important; width: 100% !important; font-size: 0.82rem !important; border-radius: 10px; overflow: hidden; }
.stMarkdown th  { background: #fdf0e8 !important; color: #c4622d !important; padding: 9px 13px !important; border: 1px solid #ede8e0 !important; font-weight: 600 !important; }
.stMarkdown td  { padding: 8px 13px !important; border: 1px solid #ede8e0 !important; color: #3a3028 !important; }
.stMarkdown tr:nth-child(even) td { background: #faf8f4 !important; }

/* ── Expander ── */
[data-testid="stExpander"] { background: #fff !important; border: 1px solid #ede8e0 !important; border-radius: 10px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important; }
[data-testid="stExpander"] summary { color: #7a6e62 !important; font-size: 0.82rem !important; }

/* ── Sidebar ── */
[data-testid="stSlider"] label { color: #7a6e62 !important; font-size: 0.8rem !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #c4622d !important; border-color: #c4622d !important; }
[data-testid="stCheckbox"] label { color: #7a6e62 !important; font-size: 0.82rem !important; }
.sidebar-section {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #c0b4a4; margin: 1.4rem 0 0.6rem;
}

/* ── Alert / Spinner ── */
[data-testid="stAlert"] { border-radius: 10px !important; font-family: 'Outfit', sans-serif !important; font-size: 0.84rem !important; }
[data-testid="stSpinner"] > div { border-top-color: #c4622d !important; }

/* ═══════════════════════════════════════════════════
   RESPONSIVE — Tablet (≤ 1024px)
═══════════════════════════════════════════════════ */
@media screen and (max-width: 1024px) {
    .block-container {
        padding: 1.5rem 1.5rem 3rem !important;
    }
    .navbar-wordmark { font-size: 1.7rem !important; }
    .navbar-sub { font-size: 0.65rem !important; }
    [data-testid="stHorizontalBlock"] { gap: 1.2rem !important; }
}

/* ═══════════════════════════════════════════════════
   RESPONSIVE — Mobile (≤ 768px)
═══════════════════════════════════════════════════ */
@media screen and (max-width: 768px) {
    /* Tighter padding on mobile */
    .block-container {
        padding: 1rem 1rem 3rem !important;
    }

    /* Navbar: smaller logo, hide subtitle */
    .navbar {
        padding-bottom: 1.2rem !important;
        margin-bottom: 1.5rem !important;
    }
    .navbar-wordmark { font-size: 1.5rem !important; }
    .navbar-sub { display: none !important; }
    .navbar-pill {
        font-size: 0.6rem !important;
        padding: 3px 10px !important;
    }

    /* Stack columns vertically on mobile */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important;
        min-width: 100% !important;
        flex: none !important;
    }

    /* Left column bottom border when stacked */
    [data-testid="stHorizontalBlock"] > div:first-child {
        border-bottom: 1.5px solid #e8e2d6 !important;
        padding-bottom: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* Section header smaller */
    .section-header { font-size: 0.6rem !important; margin-bottom: 0.8rem !important; }

    /* Product cards: no hover lift on touch */
    .product-card:hover { transform: none !important; }
    .product-name { font-size: 0.82rem !important; }

    /* Chat input full width */
    [data-testid="stChatInput"] { border-radius: 10px !important; }

    /* Info/empty box padding */
    .info-box { padding: 1.5rem 1rem !important; }

    /* Suggestion chips: 2 columns on mobile */
    [data-testid="stHorizontalBlock"] .chip-btn [data-testid="stButton"] > button {
        font-size: 0.72rem !important;
        padding: 0.4rem 0.6rem !important;
    }

    /* Markdown tables: allow horizontal scroll */
    .stMarkdown { overflow-x: auto !important; }
    .stMarkdown table { min-width: 400px !important; font-size: 0.76rem !important; }

    /* Status pill smaller */
    .status-pill { font-size: 0.7rem !important; padding: 4px 10px !important; }

    /* Buttons full-width naturally on mobile already */
    [data-testid="stButton"] > button {
        font-size: 0.84rem !important;
        padding: 0.6rem 1.2rem !important;
    }
}

/* ═══════════════════════════════════════════════════
   RESPONSIVE — Small Phone (≤ 480px)
═══════════════════════════════════════════════════ */
@media screen and (max-width: 480px) {
    .block-container { padding: 0.8rem 0.8rem 2rem !important; }

    .navbar-wordmark { font-size: 1.3rem !important; }
    .navbar-pill { display: none !important; }
    .navbar { padding-bottom: 1rem !important; margin-bottom: 1.2rem !important; }

    .product-card { padding: 12px 13px !important; }
    .product-name { font-size: 0.8rem !important; }
    .product-price { font-size: 0.95rem !important; }
    .product-badge { font-size: 0.62rem !important; }

    [data-testid="stFileUploader"] > div { padding: 1.2rem 1rem !important; }

    .info-box { padding: 1.2rem 0.9rem !important; font-size: 0.8rem !important; }

    /* Chips: single column on very small screens */
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
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.3rem;font-weight:600;
                    color:#1a1814;margin-bottom:1.5rem;padding-bottom:1rem;
                    border-bottom:1px solid #e8e2d6;letter-spacing:-0.01em">
            Shop<span style="color:#c4622d">Lens</span> AI
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
        <div class="info-box" style="background:#fdf8f4;border-color:#e8ddd0">
            <strong style="color:#1a1814;font-weight:600">{len(PRODUCT_CATALOG)} products</strong><br>
            <span style="font-size:0.76rem;color:#9b8e7e">{' · '.join(cats)}</span>
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
        <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #e8e2d6;
                    font-size:0.68rem;color:#c0b4a4;line-height:2">
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
            <span class="navbar-sub">E-Commerce Product Assistant</span>
        </div>
        <span class="navbar-pill">Multimodal RAG</span>
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
                <div style="font-size:0.82rem;color:#5a5048;line-height:1.75;font-family:'Outfit',sans-serif">
                    {st.session_state.image_desc}
                </div>""", unsafe_allow_html=True)

        if st.session_state.image_ready and st.session_state.chat_session.current_products:
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            render_products(st.session_state.chat_session.current_products[:top_k])

    # ── RIGHT ─────────────────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)

        if not st.session_state.image_ready:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:4rem 2.5rem;background:#fff">
                <div style="font-size:3rem;margin-bottom:1.2rem;line-height:1">🛍️</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:1.5rem;
                            font-weight:600;color:#1a1814;margin-bottom:0.6rem;letter-spacing:-0.01em">
                    Upload a product image
                </div>
                <div style="font-size:0.84rem;color:#9b8e7e;line-height:1.8;max-width:320px;margin:0 auto">
                    Drop any product photo — sneakers, laptops, phones, clothing.
                    Ask about specs, find alternatives, and compare prices.
                </div>
                <div style="margin-top:1.8rem;display:flex;justify-content:center;gap:10px;flex-wrap:wrap">
                    <span style="font-size:0.72rem;padding:4px 12px;border-radius:12px;
                                 background:#fdf0e8;color:#c4622d;border:1px solid #edc4a4">
                        Find similar products
                    </span>
                    <span style="font-size:0.72rem;padding:4px 12px;border-radius:12px;
                                 background:#f0f4f0;color:#2a7a4a;border:1px solid #b8dfc8">
                        Compare prices
                    </span>
                    <span style="font-size:0.72rem;padding:4px 12px;border-radius:12px;
                                 background:#f4f1eb;color:#7a6e62;border:1px solid #e0d8ce">
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
                <div style="text-align:center;padding:2.5rem 0;color:#c0b4a4;font-size:0.82rem">
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