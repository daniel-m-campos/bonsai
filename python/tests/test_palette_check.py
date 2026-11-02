"""scripts/palette_check.py: the contrast floor behind the docs palette."""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import palette_check  # noqa: E402


def test_wcag_contrast_matches_the_published_extremes():
    assert palette_check.wcag_contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert palette_check.wcag_contrast("#ffffff", "#ffffff") == pytest.approx(1.0)


def test_apca_sign_follows_polarity():
    """Dark text on light reads positive, light text on dark negative, and
    the reference pair #888 on white lands near the published Lc 63."""
    assert palette_check.apca_lc("#888888", "#ffffff") == pytest.approx(63.1, abs=0.5)
    assert palette_check.apca_lc("#ffffff", "#888888") < 0


def test_the_committed_palette_clears_the_floor():
    css = palette_check.CSS.read_text()
    for scheme in palette_check.SCHEMES:
        assert palette_check.misses(palette_check.scheme_tokens(css, scheme)) == []


def test_a_low_contrast_token_is_named():
    css = palette_check.CSS.read_text()
    tokens = palette_check.scheme_tokens(css, "bonsai-light")
    tokens["ink-muted"] = tokens["ground"]
    bad = palette_check.misses(tokens)
    assert any(line.startswith("ink-muted on ground: WCAG 1.00") for line in bad)
