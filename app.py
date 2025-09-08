import io
import os
import time
import requests
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from pypdf import PdfReader
import pyarrow as pa
import json

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


def stream_chat(messages, model: str = GEN_MODEL) -> Tuple[str, Dict]:
    """Stream chat responses with statistics."""
    response_text = ""
    token_count = 0
    start_time = time.time()
    first_token_time = None
    
    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                if content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    response_text += content
                    token_count += len(content.split())
                    yield content
    except Exception as e:
        yield f"\nError: {e}"
    
    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    time_to_first_token = (first_token_time - start_time) if first_token_time else 0
    
    stats = {
        "total_time": total_time,
        "time_to_first_token": time_to_first_token,
        "tokens": token_count,
        "tokens_per_sec": token_count / total_time if total_time > 0 else 0
    }
    
    return response_text, stats


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
if 'response_stats' not in st.session_state:
    st.session_state.response_stats = []

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
        
        # Ollama prerequisite warning
        st.warning("""
        ⚠️ **Prerequisites Required:**
        1. Install Ollama: `brew install ollama`
        2. Start Ollama: `ollama serve`
        3. Pull required models:
           - `ollama pull embeddinggemma:300m`
           - `ollama pull gpt-oss:20b`
        """)
        
        embed_model = st.text_input("Embedding Model", value=EMBED_MODEL,
                                   help="Make sure this model is installed in Ollama")
        gen_model = st.text_input("Generation Model", value=GEN_MODEL,
                                 help="Make sure this model is installed in Ollama")
        
        # Check Ollama connection
        if st.button("🔍 Check Ollama Status"):
            try:
                # Try to list models
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    if 'models' in models_data and models_data['models']:
                        available_models = [m['name'] for m in models_data['models']]
                        st.success(f"✅ Ollama is running! Available models: {', '.join(available_models)}")
                        
                        # Check if required models are installed
                        model_names = ' '.join(available_models).lower()
                        embed_found = any(embed_model.lower() in model.lower() for model in available_models)
                        gen_found = any(gen_model.lower() in model.lower() for model in available_models)
                        
                        if not embed_found:
                            st.error(f"❌ Embedding model '{embed_model}' not found. Please run: `ollama pull {embed_model}`")
                        else:
                            st.info(f"✅ Embedding model '{embed_model}' is available")
                            
                        if not gen_found:
                            st.error(f"❌ Generation model '{gen_model}' not found. Please run: `ollama pull {gen_model}`")
                        else:
                            st.info(f"✅ Generation model '{gen_model}' is available")
                    else:
                        st.warning("⚠️ Ollama is running but no models are installed. Please pull the required models.")
                else:
                    st.error(f"❌ Cannot connect to Ollama. Status code: {response.status_code}")
            except requests.ConnectionError:
                st.error("❌ Cannot connect to Ollama. Make sure it's running with: `ollama serve`")
            except Exception as e:
                st.error(f"❌ Error checking Ollama status: {e}")

# Main chat interface
st.header("💬 Ask Questions")

# Show statistics for last response if available
if st.session_state.response_stats:
    last_stats = st.session_state.response_stats[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Response Time", f"{last_stats['total_time']:.2f}s")
    with col2:
        st.metric("🚀 First Token", f"{last_stats['time_to_first_token']:.2f}s")
    with col3:
        st.metric("📝 Tokens", last_stats['tokens'])
    with col4:
        st.metric("⚡ Speed", f"{last_stats['tokens_per_sec']:.1f} tok/s")
    st.divider()

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show stats for assistant messages
        if message["role"] == "assistant" and i < len(st.session_state.response_stats):
            with st.expander("📊 Response Metrics", expanded=False):
                stats = st.session_state.response_stats[i // 2] if i // 2 < len(st.session_state.response_stats) else None
                if stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Total time: {stats['total_time']:.2f}s")
                        st.caption(f"First token: {stats['time_to_first_token']:.2f}s")
                    with col2:
                        st.caption(f"Tokens: {stats['tokens']}")
                        st.caption(f"Speed: {stats['tokens_per_sec']:.1f} tokens/sec")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        # Create placeholder for status updates
        status_placeholder = st.empty()
        
        # Phase 1: Search
        search_start = time.time()
        with status_placeholder.container():
            with st.spinner("🔍 Searching through documents..."):
                search_results = search_similar(prompt, table, k=top_k, model=embed_model)
        search_time = time.time() - search_start
        
        if search_results is not None and not search_results.empty:
            # Prepare context
            system_prompt, user_prompt, citations = prepare_context(prompt, search_results)
            
            # Show search time and sources
            status_placeholder.success(f"✅ Found {len(search_results)} relevant chunks in {search_time:.2f}s")
            
            with st.expander("📖 Sources", expanded=True):
                st.markdown(citations)
                st.caption(f"Search completed in {search_time:.2f} seconds")
            
            # Phase 2: Generate response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Create containers for response and metrics
            response_container = st.empty()
            metrics_container = st.empty()
            
            # Stream response with live metrics
            response_text = ""
            token_count = 0
            start_time = time.time()
            first_token_time = None
            
            with response_container.container():
                with st.spinner("🤔 Thinking..."):
                    time.sleep(0.5)  # Brief pause for better UX
            
            response_placeholder = st.empty()
            
            try:
                stream = ollama.chat(model=gen_model, messages=messages, stream=True)
                
                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                                status_placeholder.empty()  # Clear status once streaming starts
                            
                            response_text += content
                            token_count += len(content.split())
                            
                            # Update response
                            response_placeholder.markdown(response_text)
                            
                            # Update live metrics
                            elapsed = time.time() - start_time
                            with metrics_container.container():
                                cols = st.columns(4)
                                with cols[0]:
                                    st.caption(f"⏱️ {elapsed:.1f}s")
                                with cols[1]:
                                    st.caption(f"📝 {token_count} tokens")
                                with cols[2]:
                                    st.caption(f"⚡ {token_count/elapsed:.1f} tok/s" if elapsed > 0 else "⚡ -- tok/s")
                                with cols[3]:
                                    st.caption("🔄 Streaming...")
                
                # Final metrics
                total_time = time.time() - start_time
                ttft = (first_token_time - start_time) if first_token_time else 0
                
                stats = {
                    "total_time": total_time,
                    "time_to_first_token": ttft,
                    "tokens": token_count,
                    "tokens_per_sec": token_count / total_time if total_time > 0 else 0
                }
                
                # Update final metrics
                with metrics_container.container():
                    cols = st.columns(4)
                    with cols[0]:
                        st.caption(f"⏱️ {total_time:.2f}s total")
                    with cols[1]:
                        st.caption(f"📝 {token_count} tokens")
                    with cols[2]:
                        st.caption(f"⚡ {stats['tokens_per_sec']:.1f} tok/s")
                    with cols[3]:
                        st.caption("✅ Complete")
                
                st.session_state.response_stats.append(stats)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        else:
            no_results_msg = "I couldn't find any relevant information in the indexed documents. Please make sure documents are indexed."
            status_placeholder.warning("⚠️ No relevant documents found")
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