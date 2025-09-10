# 📚 FreeDocBuddy（免費文件夥伴）

Language Quick Start · 語言 · 语言 · Idiomas · 言語
- English: see [Quick Start (EN)](#quick-start-en)
- 繁體中文: 參見 [快速開始（繁中）](#quick-start-zh-hant)
- 简体中文: 参见 [快速开始（简中）](#quick-start-zh-hans)
- Español: ver [Inicio Rápido (ES)](#quick-start-es)
- 日本語: [クイックスタート (JA)](#quick-start-ja)

Your free, local document buddy — read, search, ask, and learn from multiple files without cloud costs or API keys.

> 零成本、本地運行、私密安全。把檔案丟進 `documents/`，就能閱讀、搜尋、問答、整理、學習。

—

Table of Contents
- [Quick Start (EN)](#quick-start-en)
- [Features](#-features)
- [Configuration (.env)](#-configuration--env)
- [Tips](#tips)
- [Languages](#languages)
- [Testing (optional)](#testing-optional)
- [Privacy & Requirements](#privacy--requirements)

## 💡 What It Does

- **Multi‑file made easy**: Drop many files in `documents/`, index once, ask anything.
- **Ask your documents**: Chat in natural language and get answers with sources.
- **Read & Learn**: Summaries, explanations, and step‑by‑step guidance from your files.
- **Search that understands**: Hybrid (keyword + vector) search for better results.
- **Local & free**: Runs entirely on your Mac with Ollama — no API keys.
- **Vision support**: Understands PDFs with images, charts, and screenshots (via LLaVA).

## 🚀 Quick Start (EN)
<a id="quick-start-en"></a>

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

<a id="quick-start-zh-hant"></a>
## 快速開始（繁中）
把檔案放進 `documents/`，索引後就能用中文/英文提問。

1) 安裝（macOS）
- `brew install ollama`
- `ollama serve &`
- `ollama pull gpt-oss:20b`
- `ollama pull embeddinggemma:300m`

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
