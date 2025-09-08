# 📚 Document Q&A System (Local RAG) — GPT-OSS-20B + EmbeddingGemma + LanceDB

A simple yet powerful document Q&A system powered by local LLMs (Ollama) that supports multiple document formats.

## ✨ Features

- **Multi-format Support**: PDF, Word, Markdown, HTML, CSV, Excel, JSON, TXT, and more
- **Auto-indexing**: Just drop files in the `documents` folder and click index
- **Local LLMs**: Uses Ollama for embeddings and generation (privacy-first)
- **Vector Search**: LanceDB for efficient similarity search  
- **Simple UI**: Clean Streamlit interface with chat history
- **Document Management**: View indexed documents, statistics, and clear index

## 🚀 Quick Start

### Prerequisites (macOS)

```bash
brew install ollama
ollama serve &
ollama pull gpt-oss:20b
ollama pull embeddinggemma:300m
```

### Setup Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## 📁 Usage

1. **Add Documents**: Drop your documents into the `documents` folder
2. **Index Documents**: Click "🔄 Index New Documents" in the sidebar
3. **Ask Questions**: Type your question in the chat interface
4. **View Sources**: Expand "📖 Sources" to see which documents were used

## 📄 Supported Formats

- **Documents**: PDF, DOCX, DOC
- **Markdown**: MD, Markdown  
- **Web**: HTML, HTM
- **Data**: CSV, XLSX, XLS, JSON
- **Text**: TXT, LOG, XML, YAML, RTF

## 🛠️ Configuration

Edit `.env` file to customize:
```env
DB_DIR=.lancedb
TABLE_NAME=docs
EMBED_MODEL=embeddinggemma:300m
GEN_MODEL=gpt-oss:20b
DOCUMENTS_FOLDER=documents
```