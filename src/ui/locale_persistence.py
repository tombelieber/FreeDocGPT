from __future__ import annotations

from pathlib import Path
from typing import Optional


def _state_dir() -> Path:
    # Place under repo root to persist across reruns
    root = Path(__file__).resolve().parents[2]
    d = root / ".app_state"
    d.mkdir(exist_ok=True)
    return d


def _locale_path() -> Path:
    return _state_dir() / "locale.txt"


def save_locale(locale: str) -> None:
    try:
        _locale_path().write_text(locale.strip(), encoding="utf-8")
    except Exception:
        # Best effort; ignore failures
        pass


def load_locale() -> Optional[str]:
    try:
        p = _locale_path()
        if p.exists():
            value = p.read_text(encoding="utf-8").strip()
            return value or None
    except Exception:
        return None
    return None

