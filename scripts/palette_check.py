"""Hold the docs palette to a contrast floor.

The site's two colour schemes live in docs/stylesheets/bonsai.css as role
tokens (`--bz-ink`, `--bz-ground`, ...). A scheme is a claim that text on
every surface stays legible, and this script is the test behind that claim:
it reads the token blocks straight from the stylesheet, computes WCAG 2
contrast and APCA lightness contrast for each text-on-surface pair the page
actually draws, and exits non-zero on any pair under the floor. It runs in
`make docs-check` after the renderers.

    python3 scripts/palette_check.py          # one line per scheme, or the misses

Floors: body ink, links and code text clear WCAG 4.5:1 and |Lc| 75 (APCA's
body-text level); muted labels clear 4.5:1 and |Lc| 60; syntax roles clear
4.5:1 on the code well. Stdlib only.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Final

REPO = pathlib.Path(__file__).resolve().parents[1]
CSS = REPO / "docs" / "stylesheets" / "bonsai.css"

SCHEMES: Final = ("bonsai-light", "bonsai-dark")
TOKEN = re.compile(r"--bz-([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;")

BODY_WCAG: Final = 4.5
BODY_LC: Final = 75.0
LABEL_LC: Final = 60.0

BODY_PAIRS: Final = (
    ("ink", "ground"),
    ("ink", "panel"),
    ("ink", "row-hover"),
    ("ink-strong", "ground"),
    ("link", "ground"),
    ("link", "panel"),
    ("link", "signal-wash"),
    ("code-text", "well"),
    ("code-text", "hl-line"),
    ("bezel-ink", "bezel"),
)
LABEL_PAIRS: Final = (
    ("ink-muted", "ground"),
    ("ink-muted", "panel"),
    ("bezel-muted", "bezel"),
    ("brass", "ground"),
    ("brass", "brass-wash"),
)
SYNTAX_ROLES: Final = (
    "hl-keyword",
    "hl-string",
    "hl-number",
    "hl-function",
    "hl-constant",
    "hl-comment",
    "hl-operator",
    "hl-special",
)


# Public Functions =================================================================================


def scheme_tokens(css: str, scheme: str) -> dict[str, str]:
    """The `--bz-*` hex tokens declared inside one scheme's block."""
    match = re.search(rf"\[data-md-color-scheme={scheme}\][^{{]*\{{(.*?)\n\}}", css, re.S)
    if match is None:
        raise SystemExit(f"palette_check: no [data-md-color-scheme={scheme}] block in {CSS}")
    return {name: value.lower() for name, value in TOKEN.findall(match.group(1))}


def wcag_contrast(fg: str, bg: str) -> float:
    """WCAG 2 contrast ratio between two hex colours."""
    lighter, darker = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def apca_lc(text: str, background: str) -> float:
    """APCA (0.0.98G-4g) lightness contrast, negative for light text on dark."""
    y_text = _soft_clamp(_luminance(text))
    y_bg = _soft_clamp(_luminance(background))
    if y_bg > y_text:
        contrast = (y_bg**0.56 - y_text**0.57) * 1.14
        return 0.0 if contrast < 0.1 else (contrast - 0.027) * 100
    contrast = (y_bg**0.65 - y_text**0.62) * 1.14
    return 0.0 if contrast > -0.1 else (contrast + 0.027) * 100


def misses(tokens: dict[str, str]) -> list[str]:
    """Every pair in one scheme that falls under its floor."""
    out: list[str] = []
    for fg, bg in BODY_PAIRS:
        out += _judge(tokens, fg, bg, BODY_WCAG, BODY_LC)
    for fg, bg in LABEL_PAIRS:
        out += _judge(tokens, fg, bg, BODY_WCAG, LABEL_LC)
    for role in SYNTAX_ROLES:
        out += _judge(tokens, role, "well", BODY_WCAG, 0.0)
    return out


def main() -> int:
    """Check both schemes; print one line per scheme or every miss."""
    css = CSS.read_text()
    failed = False
    for scheme in SCHEMES:
        bad = misses(scheme_tokens(css, scheme))
        if bad:
            failed = True
            print(f"ERROR: {scheme} fails the contrast floor:", file=sys.stderr)
            for line in bad:
                print(f"  {line}", file=sys.stderr)
            continue
        n = len(BODY_PAIRS) + len(LABEL_PAIRS) + len(SYNTAX_ROLES)
        print(f"palette {scheme}: {n} pairs clear the contrast floor")
    return 1 if failed else 0


# Private Functions ================================================================================


def _channel(value: int) -> float:
    return (value / 255) ** 2.4


def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126729 * _channel(r) + 0.7151522 * _channel(g) + 0.0721750 * _channel(b)


def _soft_clamp(y: float) -> float:
    return y if y > 0.022 else y + (0.022 - y) ** 1.414


def _judge(tokens: dict[str, str], fg: str, bg: str, wcag: float, lc: float) -> list[str]:
    if fg not in tokens or bg not in tokens:
        return [f"{fg} on {bg}: token missing"]
    ratio = wcag_contrast(tokens[fg], tokens[bg])
    contrast = abs(apca_lc(tokens[fg], tokens[bg]))
    if ratio >= wcag and contrast >= lc:
        return []
    return [f"{fg} on {bg}: WCAG {ratio:.2f} (floor {wcag}), |Lc| {contrast:.0f} (floor {lc:.0f})"]


if __name__ == "__main__":
    sys.exit(main())
