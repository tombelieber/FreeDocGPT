# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Streamlit entrypoint.
- `src/`: Library code
  - `core/`: indexing, embeddings, hybrid search, chat, caching
  - `document_processing/`: readers, analyzers, chunkers (incl. vision)
  - `config/`: `Settings`, env loading
  - `ui/`: sidebar, chat interface, settings panel
  - `utils/`: logging and Ollama helpers
- `documents/`: user-provided source files to index.
- `.lancedb/`: local vector store.
- `tests`: top-level `test_*.py` scripts.

## Build, Test, and Development Commands
- Setup env: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run app: `streamlit run app.py` (opens at `http://localhost:8501`).
- Core tests: `python test_indexing.py`, `python test_hybrid_search.py`.
- Phase 2 tests: `python test_phase2_simple.py`, `python test_phase2_features.py`.
- Vision test: `python test_vision.py` (expects a PDF in `documents/`).
- Ollama models: ensure `ollama serve` is running; models from `.env`/defaults (e.g., `gpt-oss:20b`).

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4-space indentation, 100-col soft wrap.
- Types: prefer type hints on public functions/classes.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Imports: stdlib → third-party → local; avoid circular deps (keep UI separate from `core`).

## Testing Guidelines
- Framework: lightweight Python scripts; no pytest config required.
- Naming: place quick checks in top-level `test_*.py`.
- What to run before PR: `python test_indexing.py` and `python test_hybrid_search.py` (see `CLAUDE.md`).
- Artifacts: tests may create temporary files in `documents/` or `.test_cache/`; they clean up when possible.

## Commit & Pull Request Guidelines
- Commits: prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`). Example: `feat: add Tantivy hybrid search index`.
- PRs: include purpose, linked issues, test evidence (commands + output), and screenshots for UI changes.
- Scope: keep changeset focused; update docs when changing behavior (`README.md`, `CLAUDE.md`).

## Security & Configuration Tips
- Config: copy `.env.example` → `.env`; key vars: `DB_DIR`, `TABLE_NAME`, `EMBED_MODEL`, `GEN_MODEL`, `DOCUMENTS_FOLDER`.
- Secrets: never commit real documents or credentials; verify `.gitignore` coverage.
- Local-only: system is designed to run with local Ollama; no external API keys required by default.

