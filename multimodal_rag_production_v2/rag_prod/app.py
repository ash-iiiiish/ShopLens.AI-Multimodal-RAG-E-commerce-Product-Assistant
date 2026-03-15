"""
app.py — ShopLens AI · Clean Professional UI
Matches screenshot: serif logo, cream bg, two-column, minimal
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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f5f1ea !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1a1612 !important;
}

[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── TOP BAR (thin accent line) ── */
.topbar {
    height: 3px;
    background: linear-gradient(90deg, #c4541c 0%, #e8943a 50%, #c4541c 100%);
}

/* ── NAVBAR ── */
.navbar {
    background: #f5f1ea;
    padding: 1.4rem 3.5rem;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #e2dbd0;
}
.nav-left  { display: flex; align-items: baseline; gap: 14px; }
.nav-logo  {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem; font-weight: 600; letter-spacing: -0.01em;
    color: #1a1612; line-height: 1;
}
.nav-logo span { color: #c4541c; }
.nav-tagline {
    font-size: 0.65rem; font-weight: 500; letter-spacing: 0.18em;
    text-transform: uppercase; color: #a09888;
}
.nav-badge {
    font-size: 0.62rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; padding: 5px 14px; border-radius: 100px;
    border: 1.5px solid #c4541c; color: #c4541c;
    background: transparent;
}

/* ── PIPELINE BAR ── */
.pipeline-bar {
    background: #1a1612;
    padding: 0.75rem 3.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.pb-left { display: flex; align-items: center; gap: 2rem; }
.pb-label {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.2em;
    text-transform: uppercase; color: rgba(255,255,255,0.35);
    padding-right: 2rem; border-right: 1px solid rgba(255,255,255,0.1);
}
.pb-steps { display: flex; align-items: center; gap: 6px; }
.pb-step {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.7rem; color: rgba(255,255,255,0.55); font-weight: 400;
}
.pb-step-icon {
    width: 22px; height: 22px; border-radius: 6px;
    background: rgba(255,255,255,0.07);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem;
}
.pb-arrow { color: rgba(255,255,255,0.2); font-size: 0.65rem; margin: 0 2px; }
.pb-models { display: flex; align-items: center; gap: 12px; }
.pb-model {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.68rem; color: rgba(255,255,255,0.4);
}
.pb-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }

/* ── MAIN LAYOUT ── */
.main-wrap {
    display: grid;
    grid-template-columns: 480px 1fr;
    min-height: calc(100vh - 120px);
}

/* ── LEFT PANEL ── */
.left-panel {
    background: #fff;
    border-right: 1px solid #e2dbd0;
    padding: 2.2rem 2.2rem;
}
.panel-heading {
    font-size: 0.62rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: #b0a898;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1.4rem;
}
.panel-heading::after { content: ''; flex: 1; height: 1px; background: #ede8e0; }

/* ── RIGHT PANEL ── */
.right-panel {
    background: #f5f1ea;
    padding: 2.2rem 2.5rem;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] > div {
    background: #faf8f5 !important;
    border: 1.5px dashed #cec8be !important;
    border-radius: 12px !important; padding: 2rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #c4541c !important;
    background: rgba(196,84,28,0.02) !important;
}
[data-testid="stFileUploader"] label {
    color: #8a8278 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
}
[data-testid="stFileUploader"] button {
    background: #1a1612 !important; color: #fff !important;
    border-radius: 7px !important; border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
}

/* ── ANALYZE BUTTON ── */
[data-testid="stButton"] > button {
    background: #c4541c !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.7rem 1.6rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.82rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(196,84,28,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    background: #a83e10 !important;
    box-shadow: 0 4px 18px rgba(196,84,28,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Chip buttons */
.chip-btn > button {
    background: #fff !important;
    border: 1.5px solid #c4541c !important;
    border-radius: 8px !important; color: #c4541c !important;
    font-size: 0.78rem !important; padding: 0.5rem 1rem !important;
    font-weight: 500 !important; letter-spacing: 0.01em !important;
    text-transform: none !important; box-shadow: none !important;
    transition: all 0.15s ease !important;
}
.chip-btn > button:hover {
    background: #c4541c !important; color: #fff !important;
    transform: none !important; box-shadow: none !important;
}

/* ── IMAGE ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
    border: none !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: #fff !important;
    border: 1.5px solid #ddd8ce !important; border-radius: 10px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #1a1612 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #c4541c !important;
    box-shadow: 0 0 0 3px rgba(196,84,28,0.1) !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; }

/* ── PRODUCT CARDS ── */
.product-card {
    background: #fff; border: 1px solid #e8e2d8;
    border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;
    transition: all 0.2s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.product-card:hover {
    border-color: #c4541c;
    box-shadow: 0 3px 14px rgba(196,84,28,0.1);
    transform: translateX(3px);
}
.pc-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 8px; }
.pc-name { font-size: 0.82rem; font-weight: 500; color: #1a1612; line-height: 1.35; flex: 1; }
.pc-price { font-family: 'Cormorant Garamond', serif; font-size: 1.1rem; font-weight: 600; color: #1a1612; white-space: nowrap; }
.pc-bar-wrap { height: 2px; background: #f0ece5; border-radius: 2px; margin-bottom: 8px; overflow: hidden; }
.pc-bar { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #c4541c, #e8943a); }
.pc-bottom { display: flex; align-items: center; justify-content: space-between; }
.pc-tags { display: flex; gap: 5px; }
.pctag {
    font-size: 0.62rem; font-weight: 500; padding: 2px 8px;
    border-radius: 4px; letter-spacing: 0.04em;
}
.pct-sim   { background: #fef3ec; color: #c4541c; border: 1px solid #f4c8a8; }
.pct-cat   { background: #eef3f8; color: #2a5070; border: 1px solid #b8d4e8; }
.pct-brand { background: #f4f0ea; color: #5a4830; border: 1px solid #d8caa8; }
.pc-link { font-size: 0.68rem; font-weight: 500; color: #a09888; text-decoration: none; transition: color 0.15s; }
.pc-link:hover { color: #c4541c; }

/* ── STATUS PILL ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 14px; border-radius: 6px;
    font-size: 0.74rem; font-weight: 500;
}
.sp-ready { background: #edf7f1; border: 1px solid #aad8be; color: #1a6838; }
.sp-ready .sp-dot { width: 6px; height: 6px; border-radius: 50%; background: #2aaa58; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.sp-wait { background: #f4f0ea; border: 1px solid #ddd4c4; color: #8a8070; }
.sp-wait .sp-dot { width: 6px; height: 6px; border-radius: 50%; background: #c0b8a8; }

/* ── EMPTY STATE ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center;
    padding: 3rem 2rem; min-height: 380px;
}
.empty-icon {
    width: 72px; height: 72px; border-radius: 20px;
    background: #1a1612;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; margin-bottom: 1.4rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.15);
}
.empty-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem; font-weight: 600; color: #1a1612;
    margin-bottom: 0.6rem; letter-spacing: -0.01em;
}
.empty-desc { font-size: 0.84rem; color: #9a9080; line-height: 1.8; max-width: 300px; margin-bottom: 2rem; font-weight: 300; }
.empty-tags { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
.etag {
    font-size: 0.72rem; font-weight: 500; padding: 5px 13px;
    border-radius: 6px; border: 1px solid;
}
.et-orange { background: #fef3ec; color: #c4541c; border-color: #f4c8a8; }
.et-blue   { background: #eef3f8; color: #2a5070; border-color: #b8d4e8; }
.et-green  { background: #edf7f1; color: #1a6838; border-color: #aad8be; }
.et-tan    { background: #f4f0ea; color: #5a4830; border-color: #d8caa8; }

/* ── MISC ── */
hr { border-color: #e2dbd0 !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 2px; }
.stMarkdown p { line-height: 1.8 !important; color: #2a2218 !important; font-size: 0.875rem !important; }
.stMarkdown ul, .stMarkdown ol { color: #2a2218 !important; font-size: 0.875rem !important; line-height: 1.85 !important; }
.stMarkdown strong { font-weight: 600 !important; color: #1a1612 !important; }
.stMarkdown table { border-collapse: collapse !important; width: 100% !important; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06); font-size: 0.8rem !important; }
.stMarkdown th { background: #1a1612 !important; color: #f5f1ea !important; padding: 9px 13px !important; font-weight: 600 !important; font-size: 0.72rem !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; border: none !important; }
.stMarkdown td { padding: 8px 13px !important; border-bottom: 1px solid #ede8e0 !important; color: #2a2218 !important; border-left: none !important; border-right: none !important; border-top: none !important; }
.stMarkdown tr:last-child td { border-bottom: none !important; }
.stMarkdown tr:nth-child(even) td { background: #faf8f5 !important; }
[data-testid="stExpander"] { background: #fff !important; border: 1px solid #e8e2d8 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #8a8278 !important; font-size: 0.8rem !important; }
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.84rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }
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


# ── Product cards ─────────────────────────────────────────────────────────────
def render_products(products: list):
    st.markdown("""
    <div style="font-size:0.62rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;
                color:#b0a898;display:flex;align-items:center;gap:10px;margin-bottom:1rem">
        Similar Products
        <span style="flex:1;height:1px;background:#ede8e0;display:block"></span>
    </div>
    """, unsafe_allow_html=True)
    for i, (doc, score) in enumerate(products, 1):
        m   = doc.metadata
        sim = int((1 / (1 + score)) * 100)
        st.markdown(f"""
        <div class="product-card">
            <div class="pc-top">
                <div class="pc-name">{m['name']}</div>
                <div class="pc-price">${m.get('price','N/A')}</div>
            </div>
            <div class="pc-bar-wrap">
                <div class="pc-bar" style="width:{min(sim,98)}%"></div>
            </div>
            <div class="pc-bottom">
                <div class="pc-tags">
                    <span class="pctag pct-sim">{sim}% match</span>
                    <span class="pctag pct-cat">{m.get('category','')}</span>
                    <span class="pctag pct-brand">{m.get('brand','')}</span>
                </div>
                <a class="pc-link" href="{m.get('url','#')}" target="_blank">View →</a>
            </div>
        </div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()

    # ── TOP BAR ──────────────────────────────────────────────────────────────
    st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)

    # ── NAVBAR ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="navbar">
        <div class="nav-left">
            <span class="nav-logo">Shop<span>Lens</span></span>
            <span class="nav-tagline">Product Assistant</span>
        </div>
        <span class="nav-badge">Multimodal RAG</span>
    </div>
    """, unsafe_allow_html=True)

    # ── PIPELINE BAR ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="pipeline-bar">
        <div class="pb-left">
            <span class="pb-label">Pipeline</span>
            <div class="pb-steps">
                <div class="pb-step">
                    <div class="pb-step-icon">📸</div> Image Upload
                </div>
                <span class="pb-arrow">›</span>
                <div class="pb-step">
                    <div class="pb-step-icon">👁️</div> Vision Analysis
                </div>
                <span class="pb-arrow">›</span>
                <div class="pb-step">
                    <div class="pb-step-icon">🔍</div> Hybrid Search
                </div>
                <span class="pb-arrow">›</span>
                <div class="pb-step">
                    <div class="pb-step-icon">💬</div> LLM Answer
                </div>
            </div>
        </div>
        <div class="pb-models">
            <div class="pb-model">
                <span class="pb-dot" style="background:#e86e32"></span>
                Llama 4 Scout · Vision
            </div>
            <div class="pb-model" style="padding-left:12px;border-left:1px solid rgba(255,255,255,0.08)">
                <span class="pb-dot" style="background:#7cb8f4"></span>
                LLaMA 3.3 70B · Chat
            </div>
            <div class="pb-model" style="padding-left:12px;border-left:1px solid rgba(255,255,255,0.08)">
                <span class="pb-dot" style="background:#50c898"></span>
                FAISS + BM25 · Search
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── BODY ─────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 1.4])

    # ── LEFT ─────────────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-heading">Upload Image</div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "upload", type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed"
        )

        image_source = None
        if uploaded:
            image_bytes  = uploaded.read()
            image_source = image_bytes
            st.image(Image.open(io.BytesIO(image_bytes)), use_column_width=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if image_source:
            if st.button("Analyze with Vision AI →", use_container_width=True):
                with st.spinner("Analyzing with Llama 4 Scout..."):
                    try:
                        desc = st.session_state.chat_session.load_image(image_source)
                        st.session_state.image_ready = True
                        st.session_state.image_desc  = desc
                        st.session_state.messages    = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Vision error: {e}")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if st.session_state.image_ready:
            st.markdown('<div class="status-pill sp-ready"><span class="sp-dot"></span>Image analyzed — ready to chat</div>', unsafe_allow_html=True)
            if st.session_state.image_desc:
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                with st.expander("View image analysis"):
                    st.markdown(f"""
                    <div style="font-size:0.8rem;color:#5a5048;line-height:1.85;font-weight:300">
                    {st.session_state.image_desc}</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill sp-wait"><span class="sp-dot"></span>Upload an image to begin</div>', unsafe_allow_html=True)

        if st.session_state.image_ready and st.session_state.chat_session.current_products:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            render_products(st.session_state.chat_session.current_products[:TOP_K_RESULTS])

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT ─────────────────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-heading">Conversation</div>
        """, unsafe_allow_html=True)

        if not st.session_state.image_ready:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🛍️</div>
                <div class="empty-title">Upload a product image</div>
                <div class="empty-desc">
                    Drop any product photo on the left, then ask anything
                    about it in natural language.
                </div>
                <div class="empty-tags">
                    <span class="etag et-orange">Find similar products</span>
                    <span class="etag et-blue">Compare specs</span>
                    <span class="etag et-green">Check alternatives</span>
                    <span class="etag et-tan">Best value?</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Suggestion chips — 2×2 grid to match screenshot
            suggestions = [
                "What are the features?",
                "Show alternatives under $300",
                "Compare the top 2",
                "Which is best value?"
            ]
            col1, col2 = st.columns(2)
            for i, sug in enumerate(suggestions):
                with (col1 if i % 2 == 0 else col2):
                    st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                    if st.button(sug, key=f"chip_{i}", use_container_width=True):
                        st.session_state._pending = sug
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center;padding:2.5rem 0;
                            color:rgba(0,0,0,0.2);font-size:0.8rem;">
                    Select a suggestion or type your question below
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            pending    = st.session_state.pop("_pending", None)
            user_input = st.chat_input("Ask anything about this product...") or pending

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner(""):
                        try:
                            is_compare = any(w in user_input.lower() for w in ["compare","versus","vs","difference"])
                            reply = (
                                st.session_state.pipeline.get_comparison(
                                    st.session_state.chat_session.current_products,
                                    criteria=user_input
                                ) if is_compare else
                                st.session_state.chat_session.ask(user_input)
                            )
                            st.markdown(reply)
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.error(f"Error: {e}")
                st.rerun()

            # Clear button bottom right
            if st.session_state.messages:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                if st.button("↺ Clear chat", use_container_width=False):
                    st.session_state.chat_session.clear()
                    st.session_state.messages    = []
                    st.session_state.image_ready = False
                    st.session_state.image_desc  = ""
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()