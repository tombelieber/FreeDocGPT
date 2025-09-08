import os
import io
import time
import requests
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
from pypdf import PdfReader

import ollama
import lancedb

DB_DIR = ".lancedb"
TABLE_NAME = "docs"
EMBED_MODEL = "embeddinggemma:2b"
GEN_MODEL = "gpt-oss-20b"

# ---------- Utils ----------

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end == n: break
        start = max(0, end - overlap)
    return chunks

def read_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def download_pdf(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def embed_texts(texts: List[str]) -> List[List[float]]:
    return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]

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
            "vector": lancedb.vector(1536),
        }
        tbl = db.create_table(TABLE_NAME, data=[], schema=schema)
    return tbl

def rebuild_index(docs: List[Dict[str, Any]], table):
    if not docs: return
    embs = embed_texts([d["chunk"] for d in docs])
    rows = [{"id": int(time.time()*1e6)+i, "source": d["source"], "chunk": d["chunk"], "vector": v}
            for i,(d,v) in enumerate(zip(docs, embs))]
    table.add(rows)

def search_similar(query: str, table, k: int = 5):
    qvec = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    return table.search(qvec).metric("cosine").limit(k).to_pandas()

def build_prompt(query: str, contexts: List[Dict[str, str]]) -> str:
    bullets = [f"[{i}] {c['chunk'][:500]}" for i,c in enumerate(contexts,1)]
    cites = [f"[{i}] {c['source']}" for i,c in enumerate(contexts,1)]
    system = "You are a precise research assistant. Use context only."
    user = f"Question: {query}\n\nContext:\n" + "\n".join(bullets)
    return system, user, "\n".join(cites)

def stream_chat(messages):
    stream = ollama.chat(model=GEN_MODEL, messages=messages, stream=True)
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]

# ---------- UI ----------

st.set_page_config(page_title="Agentic RAG — GPT-OSS-20B", layout="wide")
st.title("🔎 Agentic RAG — GPT-OSS-20B (Local)")

with st.sidebar:
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
                text = read_pdf_bytes(download_pdf(url.strip()))
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
            rebuild_index(docs, table)
        st.success(f"Indexed {len(docs)} chunks ✅")

st.subheader("Ask your knowledge base")
query = st.text_input("Your question")
if st.button("Ask") and query.strip():
    with st.spinner("Searching…"):
        res = search_similar(query, table, k=top_k)
    if res.empty:
        st.info("No results.")
    else:
        with st.expander("Retrieved context"):
            st.dataframe(res[["source","chunk"]])
        contexts = res.to_dict(orient="records")
        system, user, cites = build_prompt(query, contexts)
        out_box = st.empty()
        buf = ""
        for piece in stream_chat([{"role":"system","content":system},{"role":"user","content":user}]):
            buf += piece
            out_box.markdown(buf)
        st.markdown("**Sources**")
        st.code(cites)