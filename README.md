# 📚 FreeDocBuddy（免費文件夥伴）
Your free, local document buddy — read, search, ask, and learn from multiple files without cloud costs or API keys.

> 零成本、本地運行、私密安全。把檔案丟進 `documents/`，就能閱讀、搜尋、問答、整理、學習。

## 💡 What It Does

- **Multi‑file made easy**: Drop many files in `documents/`, index once, ask anything.
- **Ask your documents**: Chat in natural language and get answers with sources.
- **Read & Learn**: Summaries, explanations, and step‑by‑step guidance from your files.
- **Search that understands**: Hybrid (keyword + vector) search for better results.
- **Local & free**: Runs entirely on your Mac with Ollama — no API keys.
- **Vision support**: Understands PDFs with images, charts, and screenshots (via LLaVA).

## 🚀 Quick Start

### 1) Install prerequisites (macOS)
```bash
brew install ollama           # Local LLM runtime
ollama serve &                # Start Ollama
ollama pull gpt-oss:20b      # QA / chat model
ollama pull embeddinggemma:300m  # Embedding model for search
```

### 2) Setup Python env
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Add documents
```bash
mkdir -p documents
# Put your PDFs, Word, Markdown, HTML, CSV, TXT… here
```

### 4) Run the app
```bash
streamlit run app.py
# Open http://localhost:8501
```

### 5) Use it
- Sidebar → "🔄 Index New Documents" to index files
- Ask: “幫我總結這份合約重點？” / “Explain the API auth section.”
- See sources and tweak search settings as needed

## ✨ Features

- **Formats**: PDF, DOCX/DOC, MD, HTML, CSV, XLSX/XLS, JSON, TXT, LOG, RTF, XML, YAML
- **Presets**: Meeting notes, PRD/specs, tech docs, wiki/KB
- **Controls**: Chunk size, overlap, results count, hybrid weight
- **Metrics**: Response time, first token, tokens/sec
- **Index management**: Reset/clear index when needed

## ⚙️ Configuration (.env optional)
```env
DB_DIR=.lancedb
TABLE_NAME=docs
EMBED_MODEL=embeddinggemma:300m
GEN_MODEL=gpt-oss:20b
DOCUMENTS_FOLDER=documents
```

## Tips

- Notion/飛書(Feishu/Lark) 請匯出為 Markdown 再放進 `documents/`。
- 技術文件可把 chunk 設大一些（1500–1800），Overlap 300–400。
- 找不到答案時：多放幾份檔案或提高 Top‑K。

## Testing (optional)
- Core tests: `python test_indexing.py`, `python test_hybrid_search.py`
- Phase 2: `python test_phase2_simple.py`, `python test_phase2_features.py`
- Vision: `python test_vision.py`（需在 `documents/` 放一份 PDF）

## Privacy & Requirements
- 100% 本地運行，資料不出機器；無需雲端 API 金鑰。
- 建議 16GB+ RAM；模型下載約需 10–15GB 空間。

—

Made with love for everyone who wants AI‑powered document help without paying for the cloud.
