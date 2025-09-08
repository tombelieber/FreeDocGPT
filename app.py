import io
import os
import time
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse
from dotenv import load_dotenv

import streamlit as st
from pypdf import PdfReader

import ollama
import lancedb

# Load environment variables
load_dotenv()

DB_DIR = os.getenv("DB_DIR", ".lancedb")
TABLE_NAME = os.getenv("TABLE_NAME", "docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma:300m")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-oss:20b")

# ---------- Utils ----------


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def read_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join([p.extract_text() or "" for p in reader.pages])


def download_pdf(url: str) -> bytes:
    # Validate URL to prevent SSRF attacks
    parsed = urlparse(url)
    if parsed.scheme not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    if not parsed.netloc:
        raise ValueError("Invalid URL: missing domain")

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    try:
        return [ollama.embeddings(model=model, prompt=t)["embedding"] for t in texts]
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return []


def get_db_table(reset: bool = False):
    db = lancedb.connect(DB_DIR)
    if reset and TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    try:
        tbl = db.open_table(TABLE_NAME)
    except Exception:
        schema = {
            "id": int,
            "source": str,
            "chunk": str,
            "vector": lancedb.vector(768),  # Fixed: dimension matches embeddinggemma:300m
        }
        tbl = db.create_table(TABLE_NAME, data=[], schema=schema)
    return tbl


def rebuild_index(docs: List[Dict[str, Any]], table, model: str = EMBED_MODEL):
    if not docs:
        return
    embs = embed_texts([d["chunk"] for d in docs], model)
    if not embs:  # Handle embedding failure
        return
    rows = [{"id": int(time.time()*1e6)+i, "source": d["source"], "chunk": d["chunk"], "vector": v}
            for i, (d, v) in enumerate(zip(docs, embs))]
    table.add(rows)


def search_similar(query: str, table, k: int = 5, model: str = EMBED_MODEL):
    try:
        qvec = ollama.embeddings(model=model, prompt=query)["embedding"]
        return table.search(qvec).metric("cosine").limit(k).to_pandas()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return None


def build_prompt(query: str, contexts: List[Dict[str, str]]) -> str:
    bullets = [f"[{i}] {c['chunk'][:500]}" for i, c in enumerate(contexts, 1)]
    cites = [f"[{i}] {c['source']}" for i, c in enumerate(contexts, 1)]
    system = "You are a precise research assistant. Use context only."
    user = f"Question: {query}\n\nContext:\n" + "\n".join(bullets)
    return system, user, "\n".join(cites)


def stream_chat(messages, model: str = GEN_MODEL):
    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    except Exception as e:
        yield f"\nError: {e}"


# ---------- UI ----------


st.set_page_config(page_title="Agentic RAG — GPT-OSS-20B", layout="wide")
st.title("🔎 Agentic RAG — GPT-OSS-20B (Local)")

with st.sidebar:
    st.header("Models")
    # Allow users to override models in UI
    embed_model = st.text_input("Embedding Model", value=EMBED_MODEL, 
                                help="Ollama embedding model name")
    gen_model = st.text_input("Generation Model", value=GEN_MODEL,
                             help="Ollama chat model name")
    
    st.header("Ingest")
    pdf_urls = st.text_area("PDF URLs (one per line)")
    uploaded = st.file_uploader("Or upload PDFs", accept_multiple_files=True, type=["pdf"])
    chunk_chars = st.slider("Chunk size", 500, 2400, 1200, step=100)
    overlap = st.slider("Overlap", 0, 400, 200, step=50)
    top_k = st.slider("Top-k", 1, 10, 5)
    reset_index = st.checkbox("Rebuild Index", value=False)
    build = st.button("Build Index")

table = get_db_table(reset=reset_index)

if build:
    docs = []
    if pdf_urls.strip():
        for url in pdf_urls.splitlines():
            try:
                url = url.strip()
                if not url:
                    continue
                text = read_pdf_bytes(download_pdf(url))
                for c in chunk_text(text, chunk_chars, overlap):
                    docs.append({"source": url, "chunk": c})
            except Exception as e:
                st.error(f"Failed {url}: {e}")
    if uploaded:
        for f in uploaded:
            try:
                text = read_pdf_bytes(f.read())
                for c in chunk_text(text, chunk_chars, overlap):
                    docs.append({"source": f.name, "chunk": c})
            except Exception as e:
                st.error(f"Failed {f.name}: {e}")
    if docs:
        with st.spinner("Embedding & indexing…"):
            rebuild_index(docs, table, embed_model)
        st.success(f"Indexed {len(docs)} chunks ✅")

st.subheader("Ask your knowledge base")
query = st.text_input("Your question")
if st.button("Ask") and query.strip():
    with st.spinner("Searching…"):
        res = search_similar(query, table, k=top_k, model=embed_model)
    if res is None or res.empty:
        st.info("No results.")
    else:
        with st.expander("Retrieved context"):
            st.dataframe(res[["source", "chunk"]])
        contexts = res.to_dict(orient="records")
        system, user, cites = build_prompt(query, contexts)
        out_box = st.empty()
        buf = ""
        for piece in stream_chat([{"role": "system", "content": system}, {"role": "user", "content": user}], gen_model):
            buf += piece
            out_box.markdown(buf)
        st.markdown("**Sources**")
        st.code(cites)
