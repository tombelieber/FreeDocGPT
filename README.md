# 📚 FreeDocGPT（免費文件GPT）

Language Quick Start · 語言 · 语言 · Idiomas · 言語
- English: see [Quick Start (EN)](#quick-start-en)
- 繁體中文: 參見 [快速開始（繁中）](#quick-start-zh-hant)
- 简体中文: 参见 [快速开始（简中）](#quick-start-zh-hans)
- Español: ver [Inicio Rápido (ES)](#quick-start-es)
- 日本語: [クイックスタート (JA)](#quick-start-ja)

Your free, local document AI assistant — read, search, ask, and learn from multiple files without cloud costs or API keys.

> 零成本、本地運行、私密安全。把檔案丟進 `documents/`，就能閱讀、搜尋、問答、整理、學習。

With local models like gpt-oss:20b and Embedding Gemma available in Ollama, you can run this AI document helper fully on your Mac — no cloud, no sign‑ups. Just drop files and ask.

## 💬 Community & Support

Join our Discord community for support, discussions, and updates: [https://discord.gg/usRtaeY8](https://discord.gg/usRtaeY8)

- 🤝 Get help from the community
- 🐛 Report issues and bugs
- 💡 Share feature ideas and feedback  
- 📢 Stay updated on new releases

—

Table of Contents
- [Why This Project & Why Now](#-why-this-project--why-now)
- [For Everyone (No Tech Needed)](#-for-everyone-no-tech-needed)
- [For Technical Users](#-for-technical-users)
- [Quick Start (EN)](#quick-start-en)
- [Features](#-features)
- [Configuration (.env)](#-configuration--env)
- [Tips](#tips)
- [Languages](#languages)
- [Testing (optional)](#testing-optional)
- [Privacy & Requirements](#privacy--requirements)

## 🌍 Why This Project & Why Now

Open‑source model releases make private, on‑your‑Mac AI document help possible for everyone — even if you're not technical:

- `gpt-oss:20b` can answer questions clearly and fluently.
- Embedding Gemma helps the app quickly "remember" what's in your files.
- With Ollama, these run on your computer — no cloud accounts, no fees, and your files never leave your device.

Why we built FreeDocGPT:
- So you don't miss this moment just because it used to be "too technical".
- A friendly app: drop files, press "Index", then ask questions in plain language.
- Sensible defaults baked in; you don't need to know how it works inside.

In short: Modern local models make trustworthy, private document help possible at home. FreeDocGPT puts it one click away.

## 👋 For Everyone (No Tech Needed)

- Put your files in the `documents/` folder (PDF, Word, Markdown, etc.).
- Click “🔄 Index New Documents” in the sidebar.
- Ask questions like “Summarize this contract” or “Where is the API auth section?”.
- See the original sources used for the answer.
- Everything stays on your Mac; after the first run, it can work offline.

Tip: If you see unfamiliar terms, ignore them — the default settings already give good results.

## 🚀 Quick Start (EN)
<a id="quick-start-en"></a>

### 1) Install prerequisites (macOS)
```bash
brew install ollama           # Local LLM runtime
ollama serve &                # Start Ollama
ollama pull gpt-oss:20b      # QA / chat model
ollama pull embeddinggemma:300m  # Embedding model for search
ollama pull llava:7b         # (Optional) Vision model for image Q&A
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

## 🧑‍🔬 For Technical Users

The sections below explain how the app is built and how to tune it. If you’re not technical, you can skip to Quick Start and be fine.

## 🏗️ Architecture & Techniques

This app is a local, privacy‑first RAG (Retrieval‑Augmented Generation) system optimized for Apple Silicon (M‑series) Macs.

- Hybrid Retrieval: BM25 (Tantivy) + Vector (LanceDB) fused via Reciprocal Rank Fusion (RRF).
- Embeddings: Local Ollama model (default `embeddinggemma:300m`).
- Generation: Local Ollama model (default `gpt-oss:20b`).
- Reranking (optional): Cross‑encoder models via `sentence-transformers` for higher relevance.
- Chunking: Character‑based by default; token‑precise chunking via Tiktoken + spaCy available.
- Vision: PDF text + images + tables via PyMuPDF, pdfplumber; image understanding via LLaVA (pull `llava:7b`).
- Indexing: Incremental (xxHash change detection), content + similarity dedup (Faiss), rich metadata.
- Caching: Utilities for disk‑based caches (diskcache) for embeddings and search results are included.
- Async/Batch: Async/batch modules using asyncio + executors are included.

Why it’s fast and local:
- Tantivy (Rust) for BM25; LanceDB + PyArrow for vectors and metadata.
- ARM64‑friendly libs (xxHash, Faiss‑CPU, Tiktoken) tuned for M‑series.

## 🧭 Project Structure

```
app.py                  # Streamlit entrypoint
src/
  core/                 # Retrieval, indexing, chat, caching, hybrid
    database.py         # LanceDB + PyArrow schema and access
    indexer.py          # Readers → chunkers → embeddings → DB (+hybrid)
    hybrid_search.py    # Tantivy BM25 + Vector + RRF fusion
    embeddings.py       # Ollama embeddings
    search.py           # Search service (hybrid/vector, optional rerank)
    reranker.py         # Cross‑encoder reranking (sentence‑transformers)
    query_expansion.py  # Synonyms/abbreviations/variations for recall
    cache.py            # Embedding + search result caching (diskcache)
    async_processor.py  # Async read/batch embedding/parallel search
    token_chunker.py    # Token‑aware + sentence/code‑aware chunking
    deduplication.py    # xxHash content + Faiss similarity dedup
    chat.py             # LLM chat streaming with metrics
    vision_chat.py      # LLaVA‑assisted visual Q&A for PDFs/images
  document_processing/
    readers.py          # PDF/DOCX/MD/HTML/CSV/XLSX/JSON readers
    vision_readers.py   # PDF images/tables via PyMuPDF + pdfplumber
    analyzer.py         # LLM doc type + language detection
    chunker.py          # Character + code‑aware chunking helpers
  config/
    settings.py         # Settings + .env loading
  ui/
    sidebar.py          # Indexing + search settings + language
    chat_interface.py   # Chat UI with metrics + citations
    settings_panel.py   # Presets, prompt path, Ollama checks
    modern_chat_history.py  # Enhanced chat history UI
    context_ring_widget.py  # Context visualization components
    i18n.py             # Localization utilities
  utils/
    logging_config.py   # Logging setup
    ollama_utils.py     # Ollama status/model checks
documents/              # Drop your source files here
.lancedb/               # Local vector store (auto‑created)
tests (top‑level)       # test_*.py quick scripts
```

## 🔬 How It Works

1) Ingest: Files in `documents/` are scanned recursively; readers normalize content.
2) Analyze: LLM detects doc type (meeting/prd/technical/wiki/general) and language.
3) Chunk: Character‑based or token‑precise (Tiktoken + spaCy), with overlap and code‑aware handling.
4) Dedup: Exact (xxHash) + near‑duplicate (Faiss) pruning reduces noise and storage.
5) Embed: Local embeddings via Ollama; vectors + rich metadata stored in LanceDB.
6) Hybrid Search: Tantivy BM25 and vector results combined by RRF; optional cross‑encoder reranking.
7) Generate: Context windows prepared with sources; local LLM answers with citations.
8) Vision: For PDFs, images/tables are extracted; LLaVA can be used to reason about visuals (requires an Ollama LLaVA model such as `llava:7b`).

Key metadata stored: `content_hash`, `doc_type`, `language`, `chunk_index`, `total_chunks`, `page_number`, `section_header`, `file_modified`.

## 🧪 Techniques In Detail

- Hybrid Retrieval: Tantivy BM25 keyword matching + LanceDB cosine similarity; RRF balances both signals (adjust via “Vector vs Keyword Weight”).
- Query Expansion: Synonyms/abbreviations/variations to improve recall on technical jargon.
- Reranking: Cross‑encoders (fast/balanced/accurate) re‑score top candidates for higher precision.
- Token‑Aware Chunking: Tiktoken counts ensure safe context windows; sentence/code awareness improves coherence.
- Incremental Indexing: xxHash + mtime detect changes; old chunks are removed before reindexing.
- Deduplication: xxHash exact match + Faiss similarity deduplication.
- Caching: diskcache utilities for embeddings and search results (available in code and tests; not enabled by default in the main UI).
- Async/Batch: Parallel I/O and batched embeddings modules available (not enabled by default in the main UI).

## 🎯 Who It’s For

- Developers and data/AI enthusiasts who want a local, no‑cloud RAG.
- Product/ops teams that need quick Q&A over PRDs, specs, wikis, tickets.
- Researchers and students working with PDFs, notes, and mixed‑format corpora.
- Privacy‑sensitive users (legal, finance, healthcare) who cannot upload data.

## ✅ Best Use Cases

- Technical docs and code‑adjacent content where token‑aware chunking matters.
- Product specs/PRDs and wikis where BM25 keyword hits complement semantic recall.
- Meeting notes where quick summarization and action extraction are useful.
- PDF‑heavy material (charts/figures/tables) benefiting from vision support.

## 📌 Example Scenarios

- Onboarding knowledge base: ingest handbooks, wikis, and SOPs; ask how‑to questions.
- Contract/Policy review: drop PDFs; ask for summaries, obligations, exceptions (local only).
- Research roundup: papers + blog posts; query for findings and compare sections.
- Incident retros: aggregate logs/markdown notes; search hybrids improve recall.

## 🔧 Configuration (Advanced)

Alongside the basic `.env` shown above, you can tune advanced behavior:

```env
# Search
HYBRID_SEARCH_ENABLED=true
DEFAULT_SEARCH_MODE=hybrid   # hybrid | vector | keyword
HYBRID_ALPHA=0.5             # 0=keyword‑only, 1=vector‑only
SEARCH_RESULT_LIMIT=5

# Chunking (token‑aware)
USE_TOKEN_CHUNKING=false
MAX_CHUNK_TOKENS=512
CHUNK_OVERLAP_TOKENS=50

# Reranking
USE_RERANKING=false
RERANKER_MODEL=balanced      # fast | balanced | accurate | multilingual
RERANK_TOP_K=5

# Deduplication
DEDUP_ENABLED=true
DEDUP_THRESHOLD=0.95

# UI / Locale / Prompt
DEFAULT_LOCALE=en
SYSTEM_PROMPT_PATH=rag_prompt.md

# Ollama
OLLAMA_HOST=http://localhost:11434
```

## 🧑‍💻 Run & Develop

- Setup env: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run app: `streamlit run app.py` → open `http://localhost:8501`
- Index: Sidebar → “🔄 Index New Documents”
- Tests:
  - Core: `python test_indexing.py`, `python test_hybrid_search.py`
  - Phase 2: `python test_phase2_simple.py`, `python test_phase2_features.py`
  - Vision: `python test_vision.py` (ensure a PDF exists in `documents/`)

## ⚖️ Where This Shines

- 100% local, private by design; zero API keys needed.
- Strong on mixed corpora: PRDs/specs + wikis + PDFs with images.
- Tunable retrieval: slider to balance keyword vs vector; optional reranking for precision.
- Multilingual UX: English, 繁體中文, 简体中文, Español, 日本語.

## ⚠️ When It May Not Fit

- Very large, multi‑tenant deployments needing distributed indices and auth.
- Teams requiring hosted SaaS or cross‑device sync out of the box.
- Extremely latency‑sensitive workloads without local model warm‑up.

## 🧰 Troubleshooting

- Ollama not detected: run `ollama serve` and pull models (`ollama pull embeddinggemma:300m` and `ollama pull gpt-oss:20b`). Use Settings → “Check Ollama Status”.
- LanceDB corruption: the app auto‑resets when detected; you can also click “🧹 Reset Index” in the sidebar.
- No results: make sure documents are indexed; increase Top‑K; try Hybrid mode with `alpha≈0.5`.
- Slow first answer: first token is slower on cold models; improves after warm‑up.
- Vision Q&A: works best with PDFs containing images/tables; limit very large PDFs.
  - If you see vision model errors, make sure you’ve pulled a vision model: `ollama pull llava:7b` (or set `VISION_MODEL`).

## 🧪 Tuning Tips

- Tech docs: Hybrid with alpha 0.5–0.7; Top‑K 5–7; overlap 300–400.
- Meeting notes: Keyword‑leaning (alpha 0.3–0.5); Top‑K 3–5.
- PRDs/Wikis: Balanced hybrid (alpha 0.5); chunk 1500–1800.
- Token chunking: Enable for strict context windows or code‑heavy sets.

—

## 💡 What It Does

- **Multi‑file made easy**: Drop many files in `documents/`, index once, ask anything.
- **Ask your documents**: Chat in natural language and get answers with sources.
- **Read & Learn**: Summaries, explanations, and step‑by‑step guidance from your files.
- **Search that understands**: Hybrid (keyword + vector) search for better results.
- **Local & free**: Runs entirely on your Mac with Ollama — no API keys.
 - **Vision support**: Understands PDFs with images, charts, and screenshots (via LLaVA; install a model like `llava:7b`).

<a id="quick-start-zh-hant"></a>
## 快速開始（繁中）
把檔案放進 `documents/`，索引後就能用中文/英文提問。

1) 安裝（macOS）
- `brew install ollama`
- `ollama serve &`
- `ollama pull gpt-oss:20b`
- `ollama pull embeddinggemma:300m`
- `ollama pull llava:7b`  # （選用）影像理解模型

2) 建置環境
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`

3) 放入文件：建立 `documents/` 並放入 PDF、Word、Markdown、TXT…

4) 執行：`streamlit run app.py`（開啟 http://localhost:8501）

5) 使用方式
- 側邊欄點「🔄 建立新索引」，然後在對話框提問
- 來源與搜尋設定可在頁面中查看與調整

<a id="quick-start-zh-hans"></a>
## 快速开始（简中）
把文件放进 `documents/`，建立索引后即可中/英文提问。

1) 安装（macOS）
- `brew install ollama`
- `ollama serve &`
- `ollama pull gpt-oss:20b`
- `ollama pull embeddinggemma:300m`
- `ollama pull llava:7b`  # （可选）图像理解模型

2) 环境
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`

3) 放入文件：创建 `documents/`，加入 PDF、Word、Markdown、TXT…

4) 运行：`streamlit run app.py`（打开 http://localhost:8501）

5) 用法
- 侧边栏点“🔄 建立新索引”，然后在对话框提问
- 可查看来源并调整搜索设置

<a id="quick-start-es"></a>
## Inicio Rápido (ES)
Coloca archivos en `documents/`, indexa y pregunta en español o inglés.

1) Instala (macOS)
- `brew install ollama`
- `ollama serve &`
- `ollama pull gpt-oss:20b`
- `ollama pull embeddinggemma:300m`
- `ollama pull llava:7b`  # (Opcional) modelo de visión

2) Entorno
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`

3) Añade documentos: crea `documents/` y pon PDF, Word, Markdown, TXT…

4) Ejecuta: `streamlit run app.py` (abre http://localhost:8501)

5) Uso
- Barra lateral → “🔄 Indexar nuevos documentos” y luego pregunta en el chat
- Revisa fuentes y ajusta la búsqueda

<a id="quick-start-ja"></a>
## クイックスタート (JA)
`documents/` にファイルを入れてインデックス、その後チャットで質問。

1) インストール（macOS）
- `brew install ollama`
- `ollama serve &`
- `ollama pull gpt-oss:20b`
- `ollama pull embeddinggemma:300m`
- `ollama pull llava:7b`  # （任意）画像理解モデル

2) 環境
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`

3) ドキュメント追加：`documents/` を作成し PDF / Word / Markdown / TXT… を配置

4) 実行：`streamlit run app.py`（http://localhost:8501 を開く）

5) 使い方
- サイドバー「🔄 新しいドキュメントをインデックス」で登録 → チャットで質問
- 出典の確認や検索設定の調整が可能

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
VISION_MODEL=llava:7b
```

## Tips

- Notion/飛書(Feishu/Lark) 請匯出為 Markdown 再放進 `documents/`。
- 技術文件可把 chunk 設大一些（1500–1800），Overlap 300–400。
- 找不到答案時：多放幾份檔案或提高 Top‑K。

## Languages

- Supported: English, 繁體中文 (zh‑Hant), 简体中文 (zh‑Hans), Español (es), 日本語 (ja)
- First load auto‑detects from browser language; ambiguous `zh` defaults to 繁體 (zh‑Hant)
- Change anytime via sidebar selector: “🌐 Language / 語言 / 语言”
- Optional default in `.env`: `DEFAULT_LOCALE=zh-Hant` (or `en`, `zh-Hans`, `es`, `ja`)
- URL override: add `?locale=ja` (or `en`, `zh-Hant`, `zh-Hans`, `es`)

## Testing (optional)
- Core tests: `python test_indexing.py`, `python test_hybrid_search.py`
- Phase 2: `python test_phase2_simple.py`, `python test_phase2_features.py`
- Vision: `python test_vision.py`（需在 `documents/` 放一份 PDF）

## Privacy & Requirements
- 100% 本地運行，資料不出機器；無需雲端 API 金鑰。
- 建議 16GB+ RAM；模型下載約需 10–15GB 空間。

—

Made with love for everyone who wants AI‑powered document help without paying for the cloud.
