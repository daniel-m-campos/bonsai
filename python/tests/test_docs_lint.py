"""scripts/docs_lint.py's per-file rules, one test per leg of lint_file."""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import docs_lint  # noqa: E402


def _lint(tmp_path, monkeypatch, name: str, body: str):
    """(hard, soft) from lint_file over one written fixture page."""
    monkeypatch.setattr(docs_lint, "REPO", tmp_path)
    page = tmp_path / name
    page.write_text(body)
    hard: list[tuple] = []
    soft: list[tuple] = []
    docs_lint.lint_file(page, hard, soft)
    return hard, soft


def test_em_dash_outside_code_is_hard_and_inside_is_not(tmp_path, monkeypatch):
    """Rule (a) reads prose lines and skips fenced code."""
    body = "A line with an em-dash — here.\n\n```\ncode — dash\n```\n"
    hard, soft = _lint(tmp_path, monkeypatch, "emdash.md", body)
    assert hard == [
        ("emdash.md", 1, "em-dash",
         "em-dash; use a comma, colon, or parentheses"),
    ]
    assert soft == []


def test_banned_phrase_and_word_fire_outside_code_spans(tmp_path, monkeypatch):
    """Rule (c-i) and (d) match masked text, so backticked uses stay quiet."""
    body = (
        "This is blazingly fast prose.\n"
        "\n"
        "The rung above it.\n"
        "\n"
        "A `blazingly` and a `rung` in code spans.\n"
    )
    hard, soft = _lint(tmp_path, monkeypatch, "banned.md", body)
    assert hard == [
        ("banned.md", 1, "banned-phrase", 'banned phrase "blazingly"'),
        ("banned.md", 3, "banned-word",
         'bare "rung"; name the thing: "budget", "step", or "stage"'),
    ]
    assert soft == []


def test_lib_casing_fires_in_prose_only(tmp_path, monkeypatch):
    """Rule (b) skips table rows, identifiers, paths, and dotted calls."""
    body = "\n".join([
        "We benchmark against xgboost daily.",
        "",
        "| tool | xgboost | note |",
        "",
        "The symbol xgboost_train appears.",
        "",
        "The path docs/xgboost/index holds it.",
        "",
        "The call xgboost.train returns.",
        "",
        "Also lightgbm and catboost lag.",
        "",
        "The page use/from-xgboost is a path, from-xgboost alone is not.",
    ]) + "\n"
    hard, soft = _lint(tmp_path, monkeypatch, "libs.md", body)
    assert hard == [
        ("libs.md", 1, "lib-casing", '"xgboost" in prose; write "XGBoost"'),
        ("libs.md", 11, "lib-casing", '"lightgbm" in prose; write "LightGBM"'),
        ("libs.md", 11, "lib-casing", '"catboost" in prose; write "CatBoost"'),
        ("libs.md", 13, "lib-casing", '"xgboost" in prose; write "XGBoost"'),
    ]
    assert soft == []


def test_comparative_needs_a_number_in_its_sentence(tmp_path, monkeypatch):
    """Rule (c-ii) fires per sentence and a digit anywhere in it clears."""
    body = (
        "It is significantly faster than the rest.\n"
        "\n"
        "It is significantly faster, by 2x.\n"
    )
    hard, soft = _lint(tmp_path, monkeypatch, "comparative.md", body)
    assert hard == [
        ("comparative.md", 1, "comparative",
         '"significantly faster" with no number in the sentence'),
    ]
    assert soft == []


def test_overlong_sentence_is_soft_with_its_word_count(tmp_path, monkeypatch):
    """The soft rule reports (words, file, paragraph start, sentence)."""
    filler = " ".join(["alpha", "beta", "gamma", "delta"] * 6)
    sentence = f"It is slower than the rest {filler}."
    hard, soft = _lint(tmp_path, monkeypatch, "long.md", sentence + "\n")
    assert hard == []
    assert soft == [(30, "long.md", 1, sentence)]


def test_link_dense_paragraph_skips_soft_but_not_comparative(
    tmp_path, monkeypatch,
):
    """Three or more link targets suppress the length rule alone."""
    filler = " ".join(["alpha", "beta", "gamma", "delta"] * 6)
    body = (
        "See [one](a.md) and [two](b.md) and [three](c.md): "
        f"it is much slower than the rest {filler}.\n"
    )
    hard, soft = _lint(tmp_path, monkeypatch, "dense.md", body)
    assert hard == [
        ("dense.md", 1, "comparative",
         '"much slower" with no number in the sentence'),
    ]
    assert soft == []
