"""
app.py — Streamlit UI for Multimodal RAG Shopping Assistant
Premium dark theme with professional design
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,74,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,107,74,0.08) 0%, transparent 60%),
        #0a0a0f !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1400px !important; }

/* ── Navbar ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2.5rem;
}
.navbar-brand { display: flex; align-items: center; gap: 10px; }
.navbar-logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #634aff, #ff6b4a);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.navbar-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem; font-weight: 600;
    background: linear-gradient(135deg, #fff 30%, #a89ee0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.navbar-badge {
    font-size: 0.7rem; padding: 3px 10px; border-radius: 20px;
    background: rgba(99,74,255,0.15); border: 1px solid rgba(99,74,255,0.3);
    color: #a89ee0; font-weight: 500; letter-spacing: 0.05em;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] > div {
    background: rgba(99,74,255,0.04) !important;
    border: 1.5px dashed rgba(99,74,255,0.35) !important;
    border-radius: 16px !important; padding: 1.5rem !important;
}
[data-testid="stFileUploader"] label { color: #a89ee0 !important; font-family: 'DM Sans', sans-serif !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #634aff, #8b6fff) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 1.4rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.875rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(99,74,255,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(99,74,255,0.4) !important;
}
.chip-btn > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important; color: #b8b4cc !important;
    font-size: 0.78rem !important; padding: 0.4rem 0.9rem !important;
    box-shadow: none !important; font-weight: 400 !important;
}
.chip-btn > button:hover {
    background: rgba(99,74,255,0.12) !important;
    border-color: rgba(99,74,255,0.4) !important;
    color: #c8c4e8 !important; transform: none !important; box-shadow: none !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important; padding: 0.6rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(99,74,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,74,255,0.1) !important;
}
[data-testid="stTextInput"] label { color: #7a7590 !important; font-size: 0.8rem !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,74,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(99,74,255,0.08) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; }

/* ── Product cards ── */
.product-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
    transition: all 0.2s ease; position: relative; overflow: hidden;
}
.product-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, #634aff, #ff6b4a); border-radius: 3px 0 0 3px;
}
.product-card:hover {
    background: rgba(255,255,255,0.055);
    border-color: rgba(99,74,255,0.25); transform: translateX(2px);
}
.product-rank { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; color: #634aff; text-transform: uppercase; margin-bottom: 4px; }
.product-name { font-size: 0.875rem; font-weight: 500; color: #e8e6e0; margin-bottom: 6px; line-height: 1.3; }
.product-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.product-price { font-size: 1rem; font-weight: 600; color: #7cf4b8; }
.product-badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 20px; font-weight: 500; }
.badge-match { background: rgba(99,74,255,0.15); color: #a89ee0; border: 1px solid rgba(99,74,255,0.2); }
.badge-cat   { background: rgba(255,255,255,0.06); color: #888; border: 1px solid rgba(255,255,255,0.08); }
.product-link { font-size: 0.72rem; color: #634aff; text-decoration: none; margin-top: 6px; display: inline-block; opacity: 0.7; }
.product-link:hover { opacity: 1; }

/* ── Section headers ── */
.section-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #5a5670; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.06); }

/* ── Status pills ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 500;
}
.status-ready   { background: rgba(124,244,184,0.1); border: 1px solid rgba(124,244,184,0.2); color: #7cf4b8; }
.status-waiting { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #7a7590; }

/* ── Info box ── */
.info-box {
    background: rgba(99,74,255,0.06); border: 1px solid rgba(99,74,255,0.15);
    border-radius: 12px; padding: 1rem 1.2rem;
    font-size: 0.84rem; color: #9890c4; line-height: 1.6;
}

/* ── Image ── */
[data-testid="stImage"] img { border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.08) !important; }

/* ── Misc ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
[data-testid="stHorizontalBlock"] { gap: 2rem !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(99,74,255,0.3); border-radius: 2px; }
.stMarkdown p { line-height: 1.7 !important; }
.stMarkdown table { border-collapse: collapse !important; width: 100% !important; font-size: 0.82rem !important; }
.stMarkdown th  { background: rgba(99,74,255,0.12) !important; color: #a89ee0 !important; padding: 8px 12px !important; border: 1px solid rgba(255,255,255,0.08) !important; font-weight: 500 !important; }
.stMarkdown td  { padding: 7px 12px !important; border: 1px solid rgba(255,255,255,0.06) !important; color: #c0bcd8 !important; }
.stMarkdown tr:nth-child(even) td { background: rgba(255,255,255,0.02) !important; }
[data-testid="stExpander"] { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #7a7590 !important; font-size: 0.82rem !important; }
[data-testid="stSlider"] label { color: #7a7590 !important; font-size: 0.8rem !important; }
[data-testid="stCheckbox"] label { color: #7a7590 !important; font-size: 0.82rem !important; }
.sidebar-section { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #4a4660; margin: 1.2rem 0 0.6rem; }
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
        with st.spinner("Initializing AI models..."):
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
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;color:#e8e6e0;
                    margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid rgba(255,255,255,0.06)">
            🔍 ShopLens AI
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Retrieval</div>', unsafe_allow_html=True)
        top_k = st.slider("top_k", 2, 8, TOP_K_RESULTS, label_visibility="collapsed")
        st.caption(f"Returning top **{top_k}** matches")

        st.markdown('<div class="sidebar-section">Debug</div>', unsafe_allow_html=True)
        show_desc = st.checkbox("Show image analysis", value=False)

        st.markdown('<div class="sidebar-section">Catalog</div>', unsafe_allow_html=True)
        cats = list({p["category"] for p in PRODUCT_CATALOG})
        st.markdown(f"""
        <div class="info-box">
            <strong style="color:#e8e6e0">{len(PRODUCT_CATALOG)} products</strong><br>
            <span style="font-size:0.78rem">{' · '.join(cats)}</span>
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
        <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.05);
                    font-size:0.7rem;color:#3a3650;line-height:1.8">
            LLaMA 3.3 70B · Llama 4 Scout<br>HuggingFace · FAISS · LangChain
        </div>
        """, unsafe_allow_html=True)

    return top_k, show_desc


# ── Product panel ─────────────────────────────────────────────────────────────
def render_products(products: list):
    st.markdown('<div class="section-header">Similar Products</div>', unsafe_allow_html=True)
    for i, (doc, score) in enumerate(products, 1):
        m   = doc.metadata
        sim = int((1 / (1 + score)) * 100)
        st.markdown(f"""
        <div class="product-card">
            <div class="product-rank">#{i} match</div>
            <div class="product-name">{m['name']}</div>
            <div class="product-meta">
                <span class="product-price">${m.get('price','N/A')}</span>
                <span class="product-badge badge-match">{sim}% match</span>
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
        <div class="navbar-brand">
            <div class="navbar-logo">🔍</div>
            <span class="navbar-title">ShopLens AI</span>
        </div>
        <span class="navbar-badge">MULTIMODAL RAG</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.6], gap="large")

    # ── LEFT: Image ───────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="section-header">Product Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop image here", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

        image_source = None
        if uploaded:
            image_bytes  = uploaded.read()
            image_source = image_bytes
            st.image(Image.open(io.BytesIO(image_bytes)), use_column_width=True)

        if image_source:
            if st.button("✦  Analyze Image", use_container_width=True):
                with st.spinner("Analyzing with Llama 4 Scout Vision..."):
                    try:
                        desc = st.session_state.chat_session.load_image(image_source)
                        st.session_state.image_ready = True
                        st.session_state.image_desc  = desc
                        st.session_state.messages    = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Vision error: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.image_ready:
            st.markdown('<div class="status-pill status-ready">✦ Image analyzed — ask anything</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill status-waiting">○ Waiting for image</div>', unsafe_allow_html=True)

        if show_desc and st.session_state.image_desc:
            with st.expander("Image analysis"):
                st.markdown(f'<div style="font-size:0.82rem;color:#9890c4;line-height:1.7">{st.session_state.image_desc}</div>', unsafe_allow_html=True)

        if st.session_state.image_ready and st.session_state.chat_session.current_products:
            st.markdown("<br>", unsafe_allow_html=True)
            render_products(st.session_state.chat_session.current_products[:top_k])

    # ── RIGHT: Chat ───────────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)

        if not st.session_state.image_ready:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:3.5rem 2rem">
                <div style="font-size:2.5rem;margin-bottom:1rem">🔍</div>
                <div style="color:#e8e6e0;font-weight:500;font-size:1rem;margin-bottom:0.6rem">
                    Upload a product image to start
                </div>
                <div style="font-size:0.82rem;color:#6b6880;line-height:1.7">
                    Drop any product photo — shoes, laptops, phones, clothing.<br>
                    Ask about specs, find alternatives, compare prices.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Suggestion chips
            chip_cols = st.columns(4)
            suggestions = [
                "Key features?",
                "Under $300?",
                "Compare top 2",
                "Best value?"
            ]
            for i, (col, sug) in enumerate(zip(chip_cols, suggestions)):
                with col:
                    st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                    if st.button(sug, key=f"chip_{i}", use_container_width=True):
                        st.session_state._pending = sug
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Messages
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center;padding:2rem 0;color:#3a3650;font-size:0.82rem">
                    No messages yet — use a suggestion above or type your question
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Input
            pending    = st.session_state.pop("_pending", None)
            user_input = st.chat_input("Ask about this product...") or pending

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner(""):
                        try:
                            is_compare = any(w in user_input.lower() for w in ["compare","versus","vs","difference"])
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
