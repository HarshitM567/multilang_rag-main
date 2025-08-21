import os
import glob
import streamlit as st
from src.config import settings
from src.embeddings import MultilingualEmbedder
from src.chunking import chunk_by_tokens
from src.vectordb import VectorDB
from src.translator import detect_lang, translate_text
from src.generator import generate_answer

st.set_page_config(page_title="Multi-Language RAG (Education)", layout="wide")

st.title("🌍 Multi-Language RAG — Education Domain")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Embedding model", value=settings.embedding_model_name)
    top_k = st.slider("Top-K retrieval", 1, 10, settings.top_k)
    chunk_tokens = st.slider("Chunk tokens", 100, 800, settings.chunk_tokens, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, settings.chunk_overlap, step=10)
    preferred_lang = st.text_input("Preferred answer language (ISO-639-1, e.g., en, hi, es, fr)", value="en")
    st.markdown("---")
    st.caption("Tip: add OPENAI_API_KEY / DEEPL_API_KEY in `.env` for better generation & translation.")

# Initialize components
embedder = MultilingualEmbedder(model_name)
vdb = VectorDB(settings.collection_name)

st.subheader("1) Upload & Index Documents")
uploads = st.file_uploader("Drop files (txt, md, pdf -> text-only if pre-extracted)", type=["txt", "md"], accept_multiple_files=True)
if st.button("Index uploaded files") and uploads:
    texts, metas = [], []
    for f in uploads:
        try:
            raw = f.read().decode("utf-8", errors="ignore")
        except Exception:
            raw = f.read().decode("latin-1", errors="ignore")
        # chunk
        chunks = chunk_by_tokens(raw, max_tokens=chunk_tokens, overlap=chunk_overlap)
        texts.extend([c["text"] for c in chunks])
        metas.extend([{"filename": f.name}] * len(chunks))
    if texts:
        embs = embedder.embed_texts(texts)
        vdb.add(texts, embs, metadatas=metas)
        st.success(f"Indexed {len(texts)} chunks from {len(uploads)} files.")

# Preload sample data button
if st.button("Preload sample dataset (app/data)"):
    data_files = glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.*"))
    texts, metas = [], []
    for path in data_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        chunks = chunk_by_tokens(raw, max_tokens=chunk_tokens, overlap=chunk_overlap)
        texts.extend([c["text"] for c in chunks])
        metas.extend([{"filename": os.path.basename(path)}] * len(chunks))
    if texts:
        embs = embedder.embed_texts(texts)
        vdb.add(texts, embs, metadatas=metas)
        st.success(f"Indexed {len(texts)} chunks from {len(data_files)} sample files.")

st.subheader("2) Ask a Question")
query = st.text_input("Your question (any language)")
go = st.button("Search & Answer")

if go and query.strip():
    q_lang = detect_lang(query)
    st.write(f"Detected query language: `{q_lang}`")
    q_emb = embedder.embed_text(query)
    res = vdb.query(q_emb, top_k=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    st.write("Top results:")
    for i, (d, m) in enumerate(zip(docs, metas), 1):
        with st.expander(f"Match {i} — {m.get('filename','')}"):
            st.write(d)
    # Generate answer in user's preferred language
    answer = generate_answer(query, docs, target_lang=preferred_lang)
    if preferred_lang and preferred_lang != q_lang:
        answer = translate_text(answer, preferred_lang)
    st.markdown("### Answer")
    st.write(answer)

st.markdown("---")
st.caption("Built for multi-language RAG demos. Swap vector DB or models as needed.")
