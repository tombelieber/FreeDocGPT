# Agentic RAG (Local) — GPT-OSS-20B + EmbeddingGemma + LanceDB (macOS)

This is a minimal, local-first **Agentic RAG** demo for macOS using:
- **Ollama** to run **GPT-OSS-20B** (generator) and **EmbeddingGemma** (embeddings)
- **LanceDB** as local vector DB
- **Streamlit** UI

## 0) Prerequisites (macOS)

```bash
brew install ollama
ollama serve &
ollama pull gpt-oss-20b
ollama pull embeddinggemma:2b
```

## 1) Setup Python env

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).