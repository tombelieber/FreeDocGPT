import io
import os
import time
import requests
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from pypdf import PdfReader
import pyarrow as pa
import json
import csv

import ollama
import lancedb
from docx import Document
from bs4 import BeautifulSoup
import markdown
import pandas as pd

# Load environment variables
load_dotenv()

# Configuration
DB_DIR = os.getenv("DB_DIR", ".lancedb")
TABLE_NAME = os.getenv("TABLE_NAME", "docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma:300m")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-oss:20b")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "documents")

# Supported file types
SUPPORTED_EXTENSIONS = {
    '.pdf', '.txt', '.text', '.md', '.markdown',
    '.docx', '.doc', '.html', '.htm', '.csv',
    '.xlsx', '.xls', '.json', '.xml', '.yml',
    '.yaml', '.log', '.rtf'
}

# ---------- Document Reading Functions ----------

def read_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join([p.extract_text() or "" for p in reader.pages])


def read_docx_bytes(docx_bytes: bytes) -> str:
    doc = Document(io.BytesIO(docx_bytes))
    return "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])


def read_markdown(content: str) -> str:
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator='\n', strip=True)


def read_html(content: str) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator='\n', strip=True)


def read_csv_bytes(csv_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    text_parts = []
    for _, row in df.iterrows():
        row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        if row_text:
            text_parts.append(row_text)
    return "\n".join(text_parts)


def read_excel_bytes(excel_bytes: bytes) -> str:
    dfs = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
    text_parts = []
    for sheet_name, df in dfs.items():
        text_parts.append(f"Sheet: {sheet_name}")
        for _, row in df.iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            if row_text:
                text_parts.append(row_text)
    return "\n".join(text_parts)


def read_json_bytes(json_bytes: bytes) -> str:
    data = json.loads(json_bytes.decode('utf-8'))
    return json.dumps(data, indent=2, ensure_ascii=False)


def read_file_content(file_path: Path) -> str:
    """Read content from various file formats."""
    ext = file_path.suffix.lower()
    
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        if ext == '.pdf':
            return read_pdf_bytes(file_bytes)
        elif ext in ['.docx', '.doc']:
            return read_docx_bytes(file_bytes)
        elif ext in ['.md', '.markdown']:
            content = file_bytes.decode('utf-8')
            return read_markdown(content)
        elif ext in ['.html', '.htm']:
            content = file_bytes.decode('utf-8')
            return read_html(content)
        elif ext == '.csv':
            return read_csv_bytes(file_bytes)
        elif ext in ['.xlsx', '.xls']:
            return read_excel_bytes(file_bytes)
        elif ext == '.json':
            return read_json_bytes(file_bytes)
        else:
            # Plain text files
            return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        st.error(f"Error reading {file_path.name}: {e}")
        return ""


# ---------- Core Functions ----------

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    text = " ".join(text.split())
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    try:
        embeddings = []
        progress_bar = st.progress(0)
        for i, text in enumerate(texts):
            embedding = ollama.embeddings(model=model, prompt=text)["embedding"]
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(texts))
        progress_bar.empty()
        return embeddings
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return []


def get_db_table():
    """Connect to database and get/create table."""
    db = lancedb.connect(DB_DIR)
    
    if TABLE_NAME not in db.table_names():
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("source", pa.string()),
            pa.field("chunk", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 768)),
            pa.field("timestamp", pa.string()),
        ])
        return db.create_table(TABLE_NAME, data=[], schema=schema)
    
    return db.open_table(TABLE_NAME)


def get_indexed_documents(table) -> pd.DataFrame:
    """Get list of indexed documents with statistics."""
    try:
        df = table.to_pandas()
        if df.empty:
            return pd.DataFrame()
        
        # Group by source to get document statistics
        stats = df.groupby('source').agg({
            'id': 'count',
            'timestamp': 'first'
        }).reset_index()
        stats.columns = ['Document', 'Chunks', 'Indexed At']
        stats = stats.sort_values('Indexed At', ascending=False)
        return stats
    except:
        return pd.DataFrame()


def scan_documents_folder() -> List[Path]:
    """Scan documents folder for supported files."""
    folder = Path(DOCUMENTS_FOLDER)
    if not folder.exists():
        folder.mkdir(exist_ok=True)
    
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
    
    return sorted(files)


def index_documents(files: List[Path], table, chunk_chars: int = 1200, overlap: int = 200):
    """Index multiple documents into the vector database."""
    if not files:
        st.warning("No files to index")
        return
    
    # Get already indexed files
    existing_df = get_indexed_documents(table)
    existing_sources = set(existing_df['Document'].values) if not existing_df.empty else set()
    
    # Filter out already indexed files
    new_files = [f for f in files if f.name not in existing_sources]
    
    if not new_files:
        st.info("All files are already indexed")
        return
    
    st.info(f"Indexing {len(new_files)} new document(s)...")
    
    all_docs = []
    for file_path in new_files:
        with st.spinner(f"Processing {file_path.name}..."):
            text = read_file_content(file_path)
            if text and text.strip():
                chunks = chunk_text(text, chunk_chars, overlap)
                for chunk in chunks:
                    all_docs.append({
                        "source": file_path.name,
                        "chunk": chunk,
                        "timestamp": datetime.now().isoformat()
                    })
    
    if all_docs:
        with st.spinner(f"Generating embeddings for {len(all_docs)} chunks..."):
            embeddings = embed_texts([d["chunk"] for d in all_docs], EMBED_MODEL)
            
            if embeddings:
                rows = []
                for i, (doc, emb) in enumerate(zip(all_docs, embeddings)):
                    rows.append({
                        "id": int(time.time()*1e6) + i,
                        "source": doc["source"],
                        "chunk": doc["chunk"],
                        "vector": emb,
                        "timestamp": doc["timestamp"]
                    })
                
                table.add(rows)
                st.success(f"✅ Indexed {len(new_files)} document(s) with {len(rows)} chunks")


def search_similar(query: str, table, k: int = 5, model: str = EMBED_MODEL):
    """Search for similar documents."""
    try:
        emb = ollama.embeddings(model=model, prompt=query)["embedding"]
        results = table.search(emb).limit(k).to_pandas()
        return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return None


def prepare_context(query: str, search_results) -> tuple:
    """Prepare context for the LLM."""
    if search_results is None or search_results.empty:
        return None, None, None
    
    contexts = search_results.to_dict('records')
    bullets = [f"• {c['chunk'][:500]}" for c in contexts]
    cites = [f"[{i}] {c['source']}" for i, c in enumerate(contexts, 1)]
    
    system = "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain relevant information, say so."
    user = f"Question: {query}\n\nContext:\n" + "\n".join(bullets)
    
    return system, user, "\n".join(cites)


def stream_chat(messages, model: str = GEN_MODEL):
    """Stream chat responses."""
    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    except Exception as e:
        yield f"\nError: {e}"


# ---------- UI ----------

st.set_page_config(
    page_title="📚 Document Q&A System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Q&A System")
st.markdown("Simply add documents to the `documents` folder and start asking questions!")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    
    # Display documents folder path
    st.info(f"📂 Documents folder: `./{DOCUMENTS_FOLDER}/`")
    
    # Scan for documents
    available_files = scan_documents_folder()
    
    if available_files:
        st.success(f"Found {len(available_files)} document(s)")
        
        # Index button
        if st.button("🔄 Index New Documents", type="primary", use_container_width=True):
            table = get_db_table()
            index_documents(available_files, table)
            st.rerun()
    else:
        st.warning(f"No documents found in `./{DOCUMENTS_FOLDER}/`")
        st.markdown("**Supported formats:**")
        st.markdown("PDF, Word, Markdown, HTML, CSV, Excel, JSON, TXT, etc.")
    
    st.divider()
    
    # Show indexed documents
    st.header("📊 Indexed Documents")
    table = get_db_table()
    indexed_docs = get_indexed_documents(table)
    
    if not indexed_docs.empty:
        st.dataframe(indexed_docs, use_container_width=True, hide_index=True)
        
        total_chunks = indexed_docs['Chunks'].sum()
        st.metric("Total Chunks", total_chunks)
        
        # Clear index button
        if st.button("🗑️ Clear All Index", use_container_width=True):
            db = lancedb.connect(DB_DIR)
            if TABLE_NAME in db.table_names():
                db.drop_table(TABLE_NAME)
                st.success("Index cleared!")
                st.rerun()
    else:
        st.info("No documents indexed yet")
    
    st.divider()
    
    # Settings
    with st.expander("⚙️ Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1200, step=100)
        overlap_size = st.slider("Overlap Size", 0, 400, 200, step=50)
        top_k = st.slider("Search Results", 1, 10, 5)
        
        st.subheader("Models")
        embed_model = st.text_input("Embedding Model", value=EMBED_MODEL)
        gen_model = st.text_input("Generation Model", value=GEN_MODEL)

# Main chat interface
st.header("💬 Ask Questions")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        # Search for relevant documents
        with st.spinner("Searching documents..."):
            search_results = search_similar(prompt, table, k=top_k, model=embed_model)
        
        if search_results is not None and not search_results.empty:
            # Prepare context
            system_prompt, user_prompt, citations = prepare_context(prompt, search_results)
            
            # Show sources
            with st.expander("📖 Sources"):
                st.markdown(citations)
            
            # Stream response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = st.write_stream(stream_chat(messages, model=gen_model))
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            no_results_msg = "I couldn't find any relevant information in the indexed documents. Please make sure documents are indexed."
            st.warning(no_results_msg)
            st.session_state.messages.append({"role": "assistant", "content": no_results_msg})

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    st.markdown(f"**Embedding Model:** {embed_model}")
with col3:
    st.markdown(f"**Generation Model:** {gen_model}")