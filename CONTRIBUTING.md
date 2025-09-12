# Contributing

1. Clone and create a branch from `main`.
2. Run `pytest -q` before PR.
3. Keep PRs small and focused.

## UI & Localization (i18n)

- Always localize new UI text: do not hardcode strings directly in Streamlit components. Use `t(key, default, **kwargs)` from `src/ui/i18n.py`.
- Add a translation entry for each new key to `src/ui/i18n.py` across all `SUPPORTED_LOCALES` (`en`, `zh-Hant`, `zh-Hans`, `es`, `ja`). At minimum, provide a correct English string; if other translations aren’t available, copy the English text temporarily and mark for follow‑up.
- Prefer descriptive keys (e.g., `settings.model_config_title`) and include a concise default string in the call site for readability and fallback.
- If you introduce a new locale, update `SUPPORTED_LOCALES`, `normalize_locale()`, and provide the key set for that locale.
- For prompt/UI content that is language‑specific, also update the language prompt files where applicable: `rag_prompt_en.md`, `rag_prompt_es.md`, `rag_prompt_ja.md`, `rag_prompt_zh_hans.md`, `rag_prompt_zh_hant.md`.
- Quick check before PR: run `python scripts/check_ui_i18n.py` to see warnings about hardcoded UI strings. Set `I18N_ENFORCE=1` to make it fail locally.
