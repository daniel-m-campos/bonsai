"""Mechanical STYLE.md checks over the published documentation.

The site's writing contract lives in docs/STYLE.md. This script enforces the
half of it that a machine can judge, so a regression fails CI instead of
review. Stdlib only, like the other doc generators; no build required.

Scope: the published pages only. `exclude_docs` in mkdocs.yml lists what
never reaches the site (STYLE.md itself, ops/, conversations/, reviews/, and
the loose working notes); this script parses that block and skips the same
files. It ADDITIONALLY exempts the frozen archive, docs/decisions.md,
docs/architecture/**, and docs/lineage/**: that material is a historical
record, first-class for agents and deep divers but deliberately out of the
main line, and rewriting it to today's style would falsify the record.
README.md is linted too (it is the repo's front page).

Two tiers of rule:

HARD (exit 1, one message per offending line):
  a. Em-dash anywhere outside fenced code blocks (STYLE bans it in prose).
  b. Lowercase xgboost / lightgbm / catboost in prose. Correct forms are
     XGBoost, LightGBM, CatBoost. "Prose" excludes fenced code, inline
     backtick spans, URLs and markdown link targets, table rows, and any
     token that is part of an identifier or path (an adjacent _, /, ., or
     word character, so xgboost.train and lineage/xgboost stay quiet).
  c. A tight banned-phrase list: unfalsifiable hype ("blazingly", "blazing
     fast", "clean code", "simple API", "easy to use", "world-class") and
     the comparatives "significantly/much faster|slower" when no digit
     shares their sentence (STYLE: never a comparative without a number).

SOFT (reported, exit 0): sentences over 25 words. Table rows, headings, and
link-dense lines are skipped. The top offenders print with file:line so a
later editing pass can chip at them.

    python3 scripts/docs_lint.py            # lint; exit 1 on any hard finding
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

EM_DASH = "—"
LIBS = ("xgboost", "lightgbm", "catboost")
LIB_CORRECT = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}

BANNED_SUBSTRINGS = (
    "blazingly", "blazing fast", "clean code",
    "simple api", "easy to use", "world-class",
)
COMPARATIVE_RE = re.compile(r"\b(significantly|much)\s+(faster|slower)\b", re.I)

SOFT_WORD_LIMIT = 25

FENCE_RE = re.compile(r"^\s*(```|~~~)")
LIB_RE = re.compile(r"(?<![A-Za-z0-9_/.])(" + "|".join(LIBS) + r")")


# ---- corpus selection --------------------------------------------------------

def excluded_patterns(mkdocs_text: str) -> list[str]:
    """The `exclude_docs:` block-scalar patterns, docs-relative."""
    lines = mkdocs_text.splitlines()
    pats: list[str] = []
    for idx, line in enumerate(lines):
        if not re.match(r"^exclude_docs:\s*\|", line):
            continue
        for follow in lines[idx + 1:]:
            if follow.strip() == "":
                continue
            if not follow.startswith((" ", "\t")):
                break  # dedent: next top-level key
            pats.append(follow.strip())
        break
    return pats


def is_excluded(rel: str, pats: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for p in pats:
        if p.endswith("/"):
            if rel == p.rstrip("/") or rel.startswith(p):
                return True
        elif rel == p or name == p:  # bare name matches at any depth
            return True
    return False


def is_archive(rel: str) -> bool:
    """The frozen historical record, exempt by policy (see the module docstring)."""
    return (rel == "decisions.md"
            or rel.startswith("architecture/")
            or rel.startswith("lineage/"))


def corpus_files() -> list[pathlib.Path]:
    pats = excluded_patterns((REPO / "mkdocs.yml").read_text())
    files = [REPO / "README.md"]
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(DOCS).as_posix()
        if is_excluded(rel, pats) or is_archive(rel):
            continue
        files.append(path)
    return files


# ---- text handling -----------------------------------------------------------

def classify(lines: list[str]) -> list[tuple[int, str, bool]]:
    """(lineno, text, in_code) per line; fence delimiters count as code."""
    out: list[tuple[int, str, bool]] = []
    in_code = False
    for i, line in enumerate(lines, 1):
        if FENCE_RE.match(line):
            out.append((i, line, True))
            in_code = not in_code
        else:
            out.append((i, line, in_code))
    return out


def mask_code_and_links(text: str) -> str:
    """Blank out inline code spans, link targets, and bare URLs; keep link text."""
    text = re.sub(r"``+.*?``+", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "] ", text)      # [text](url) -> keep text
    text = re.sub(r"<https?://[^>]*>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bwww\.\S+", " ", text)
    return text


def is_table_row(text: str) -> bool:
    return text.lstrip().startswith("|")


def is_heading(text: str) -> bool:
    return text.lstrip().startswith("#")


def prose_paragraphs(classified: list[tuple[int, str, bool]]):
    """Yield (start_line, joined_text) for runs of prose lines, so a hard-wrapped
    paragraph is judged as one unit (its sentences may span several lines)."""
    para: list[str] = []
    start = 0
    for lineno, text, in_code in classified:
        stripped = text.strip()
        brk = in_code or not stripped or is_table_row(text) or is_heading(text)
        if brk:
            if para:
                yield start, " ".join(para)
                para = []
        else:
            if not para:
                start = lineno
            para.append(stripped)
    if para:
        yield start, " ".join(para)


def to_plain(text: str) -> str:
    """Strip inline markdown so word counts approximate spoken words."""
    text = re.sub(r"``+.*?``+", " codespan ", text)
    text = re.sub(r"`[^`]*`", " codespan ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)         # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # links -> text
    text = re.sub(r"[*_>#]", "", text)
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def word_count(sentence: str) -> int:
    return sum(1 for w in sentence.split() if any(c.isalnum() for c in w))


# ---- rules -------------------------------------------------------------------

def lint_file(path: pathlib.Path, hard: list[tuple], soft: list[tuple]) -> None:
    rel = path.relative_to(REPO).as_posix()
    lines = path.read_text().splitlines()
    classified = classify(lines)

    for lineno, text, in_code in classified:
        if in_code:
            continue

        # (a) em-dash, anywhere outside fenced code.
        if EM_DASH in text:
            hard.append((rel, lineno, "em-dash",
                         "em-dash; use a comma, colon, or parentheses"))

        # (c-i) banned hype substrings, outside fenced code and inline code.
        masked_lower = mask_code_and_links(text).lower()
        for phrase in BANNED_SUBSTRINGS:
            if phrase in masked_lower:
                hard.append((rel, lineno, "banned-phrase",
                             f'banned phrase "{phrase}"'))

        # (b) lowercase library names in prose (not tables).
        if not is_table_row(text):
            masked = mask_code_and_links(text)
            for m in LIB_RE.finditer(masked):
                lib = m.group(1)
                nxt = masked[m.end():m.end() + 1]
                nxt2 = masked[m.end() + 1:m.end() + 2]
                if nxt.isalnum() or nxt in ("_", "/"):
                    continue  # part of an identifier or path
                if nxt == "." and nxt2.isalnum():
                    continue  # xgboost.train, file.md
                hard.append((rel, lineno, "lib-casing",
                             f'"{lib}" in prose; write "{LIB_CORRECT[lib]}"'))

    # Sentence-level rules over reconstructed prose paragraphs.
    for start, raw in prose_paragraphs(classified):
        link_dense = raw.count("](") >= 3
        plain = to_plain(raw)
        for sentence in split_sentences(plain):
            # (c-ii) comparative without a number in its sentence.
            if COMPARATIVE_RE.search(sentence) and not re.search(r"\d", sentence):
                m = COMPARATIVE_RE.search(sentence)
                hard.append((rel, start, "comparative",
                             f'"{m.group(0)}" with no number in the sentence'))
            # SOFT: overlong sentences.
            if link_dense:
                continue
            n = word_count(sentence)
            if n > SOFT_WORD_LIMIT:
                soft.append((n, rel, start, sentence))


def main() -> int:
    hard: list[tuple] = []
    soft: list[tuple] = []
    files = corpus_files()
    for path in files:
        lint_file(path, hard, soft)

    hard.sort(key=lambda h: (h[0], h[1], h[2]))
    for rel, lineno, code, msg in hard:
        print(f"{rel}:{lineno}: [{code}] {msg}")

    by_rule: dict[str, int] = {}
    for _, _, code, _ in hard:
        by_rule[code] = by_rule.get(code, 0) + 1

    print()
    print(f"docs-lint: {len(files)} files, {len(hard)} hard findings"
          + (" (" + ", ".join(f"{k}={v}" for k, v in sorted(by_rule.items())) + ")"
             if by_rule else ""))

    soft.sort(reverse=True)
    print(f"docs-lint SOFT: {len(soft)} sentences over {SOFT_WORD_LIMIT} words"
          + (" (top 10 below)" if soft else ""))
    for n, rel, lineno, sentence in soft[:10]:
        snippet = sentence if len(sentence) <= 90 else sentence[:87] + "..."
        print(f"  {n}w  {rel}:{lineno}  {snippet}")

    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
