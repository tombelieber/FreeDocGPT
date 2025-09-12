#!/usr/bin/env python3
"""
Simple i18n check for Streamlit UI code.

Flags likely hardcoded, user-visible strings in `src/ui/` that are not wrapped
with the localization helper `t(...)` from `src/ui/i18n.py`.

Heuristics:
- Look for calls to `st.*` and `st.sidebar.*` functions commonly used for UI.
- If the first positional arg or relevant keyword args (label/help/placeholder/title/caption)
  are string literals or f-strings (not a call to `t(...)`), flag them.
- Ignore lines containing inline bypass comments: `# no-i18n`, `# i18n: ignore`, `# i18n-ok`.
- Ignore emoji-only literals (no alphabetic characters).

Exit code:
- Prints findings and exits 0 by default (non-blocking).
- Set env `I18N_ENFORCE=1` to exit non-zero if issues are found.
"""
from __future__ import annotations

import ast
import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Set


UI_FUNCS: set[str] = {
    # Common display components
    "title", "header", "subheader", "markdown", "write", "caption",
    "info", "warning", "error", "success",
    # Containers / widgets
    "button", "text", "text_input", "text_area", "selectbox", "multiselect",
    "radio", "slider", "checkbox", "number_input", "date_input", "time_input",
    "file_uploader", "tabs", "expander",
}

RELEVANT_KWARGS: set[str] = {"label", "help", "placeholder", "caption", "title", "text"}

IGNORE_COMMENTS: tuple[str, ...] = ("no-i18n", "i18n: ignore", "i18n-ok")


def is_emoji_only(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    # Consider safe if no alphabetic characters present
    return not any(ch.isalpha() for ch in s)


def root_is_st(node: ast.AST) -> bool:
    """Return True if attribute chain roots at Name('st')."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id == "st"


def get_func_name(attr: ast.AST) -> str | None:
    """Return the last attribute name (e.g., 'header' for st.header)."""
    if isinstance(attr, ast.Attribute):
        return attr.attr
    return None


def is_t_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t"


def stringy(node: ast.AST) -> Tuple[bool, str | None]:
    """Return (is_stringy, constant_part) for Constant/JoinedStr nodes.
    constant_part is best-effort text for testing emoji-only filter.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True, node.value
    if isinstance(node, ast.JoinedStr):  # f-string
        # Gather literal parts if any, to test emoji-only
        parts: List[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
        return True, "".join(parts) if parts else None
    return False, None


def should_ignore_line(src_line: str) -> bool:
    comment = src_line.split("#", 1)
    if len(comment) < 2:
        return False
    trailing = comment[1].strip().lower()
    return any(tag in trailing for tag in IGNORE_COMMENTS)


def scan_file(path: Path) -> List[Tuple[int, str]]:
    """Return list of (lineno, message) for issues in given file."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [(e.lineno or 0, f"SyntaxError parsing file: {e}")]

    lines = src.splitlines()
    issues: List[Tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not root_is_st(func):
            continue
        name = get_func_name(func)
        if not name or name not in UI_FUNCS:
            continue

        # Skip ignored lines
        lineno = getattr(node, "lineno", None)
        if lineno is not None and 1 <= lineno <= len(lines):
            if should_ignore_line(lines[lineno - 1]):
                continue

        # Check first positional argument
        flagged = False
        details: List[str] = []

        if node.args:
            first = node.args[0]
            if is_t_call(first):
                pass  # ok
            else:
                is_str, const = stringy(first)
                if is_str and not (const is not None and is_emoji_only(const)):
                    flagged = True
                    details.append("positional label")

        # Check relevant keyword arguments
        for kw in node.keywords:
            if kw.arg in RELEVANT_KWARGS and kw.value is not None:
                if is_t_call(kw.value):
                    continue
                is_str, const = stringy(kw.value)
                if is_str and not (const is not None and is_emoji_only(const)):
                    flagged = True
                    details.append(f"kw:{kw.arg}")

        if flagged:
            issue = f"{name} with non-localized string ({', '.join(details)})"
            issues.append((lineno or 0, issue))

    return issues


def find_ui_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # Skip the i18n catalog itself
        if p.name == "i18n.py":
            continue
        yield p


def load_baseline(path: Path) -> Set[tuple[str, int]]:
    items: Set[tuple[str, int]] = set()
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        p, ln = line.rsplit(":", 1)
        try:
            items.add((p, int(ln)))
        except ValueError:
            continue
    return items


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except Exception:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check UI files for non-localized strings.")
    parser.add_argument("--baseline", type=str, default=os.environ.get("I18N_BASELINE", ""), help="Path to baseline file of known issues to ignore.")
    parser.add_argument("--write-baseline", type=str, default="", help="Write current findings to this baseline file and exit 0.")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero if issues are found (after baseline filtering).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ui_dir = repo_root / "src" / "ui"
    if not ui_dir.exists():
        print("No src/ui directory found; skipping i18n check.")
        return 0

    baseline_items: Set[tuple[str, int]] = set()
    baseline_path: Path | None = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_absolute():
            baseline_path = repo_root / baseline_path
        baseline_items = load_baseline(baseline_path)

    findings: list[tuple[str, int, str]] = []  # (rel_path, line, message)
    for file in sorted(find_ui_files(ui_dir)):
        issues = scan_file(file)
        rel = repo_relative(file, repo_root)
        for lineno, msg in issues:
            if (rel, lineno) in baseline_items:
                continue
            findings.append((rel, lineno, msg))

    if args.write_baseline:
        out = Path(args.write_baseline)
        if not out.is_absolute():
            out = repo_root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        # Write full set BEFORE baseline filtering to capture all current issues
        all_issues: list[tuple[str, int]] = []
        for file in sorted(find_ui_files(ui_dir)):
            rel = repo_relative(file, repo_root)
            for lineno, _ in scan_file(file):
                all_issues.append((rel, lineno))
        with out.open("w", encoding="utf-8") as f:
            f.write("# i18n UI baseline — known locations of hardcoded strings\n")
            f.write("# Format: <path>:<line>\n")
            for p, ln in sorted(all_issues):
                f.write(f"{p}:{ln}\n")
        print(f"Baseline written to {out}")
        return 0

    if not findings:
        print("i18n check: OK — no hardcoded UI strings detected (after baseline).")
        return 0

    # Human-friendly report
    grouped: dict[str, list[tuple[int, str]]] = {}
    for p, ln, msg in findings:
        grouped.setdefault(p, []).append((ln, msg))
    for p in sorted(grouped):
        print(f"\n[file] {p}")
        for ln, msg in sorted(grouped[p]):
            print(f"  L{ln}: {msg}")
    print(f"\ni18n check: {len(findings)} potential hardcoded UI strings found (new or outside baseline).")

    enforce_env = os.environ.get("I18N_ENFORCE", "0").lower() not in ("", "0", "false")
    enforce = args.enforce or enforce_env
    return 1 if enforce else 0


if __name__ == "__main__":
    raise SystemExit(main())
