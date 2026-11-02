"""docs/hooks/home_readout.py: the landing hero is the scenario panel's
headline, and its door counts are read from the sources they count."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import render_results  # noqa: E402


def _load_hook():
    spec = importlib.util.spec_from_file_location(
        "home_readout", REPO / "docs/hooks/home_readout.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


home_readout = _load_hook()


NAV = [
    {"Home": "index.md"},
    {
        "Learn": [
            {"The algorithm": ["guide/README.md", {"0. A tree by hand": "guide/0.md"}]},
            {"The engine": [{"E1. The marginal round": "learn/engine/1.md"}]},
            {"The CPU track": [{"C1. Cache awareness": "learn/cpu/1.md"}]},
            {"How we got here": "learn/timeline.md"},
        ]
    },
]


def _live_nav() -> list:
    yaml = pytest.importorskip("yaml")
    text = (REPO / "mkdocs.yml").read_text()
    return yaml.safe_load(text[text.index("nav:") :])["nav"]


def test_readout_rows_follow_the_headline_table():
    """Every grower has both planes, each arm's bar is its share of the
    slower arm, and the bold arm is the one the headline table bolds."""
    readout = home_readout.readout()
    table = render_results.summary_matrix()
    assert [s["key"] for s in readout["scenarios"]] == ["tall", "wide", "extreme"]
    tall = readout["scenarios"][0]
    assert [row["grower"] for row in tall["rows"]] == ["depthwise", "leafwise", "levelwise"]
    for row in tall["rows"]:
        line = next(line for line in table.splitlines() if line.startswith(f"| {row['grower']} "))
        for cell in row["cells"]:
            bonsai, rival = cell["arms"]
            assert bonsai["who"] == "bonsai"
            assert 0.0 < float(bonsai["w"]) < float(rival["w"]) <= 1.0
            assert not (bonsai["wins"] and rival["wins"])
            if bonsai["wins"]:
                assert f"**bonsai {bonsai['text'].replace(' ', '')}**" in line


def test_every_scenario_renders_each_arm_as_a_share_or_a_status_word():
    """Wide and extreme render like tall; bars share one scale per plane
    column; an arm that did not finish (an OOM at the extreme scenario)
    prints its status word with no bar and never wins; the extreme scenario
    has no CPU plane."""
    scenarios = home_readout.readout()["scenarios"]
    assert [len(s["planes"]) for s in scenarios] == [2, 2, 1]
    for scenario in scenarios:
        assert scenario["caption"].startswith(f"{scenario['key']} scenario · ")
        for column in range(len(scenario["planes"])):
            arms = [arm for row in scenario["rows"] for arm in row["cells"][column]["arms"]]
            finished = [arm for arm in arms if arm["finished"]]
            assert finished and max(float(arm["w"]) for arm in finished) == 1.0
            for arm in arms:
                assert arm["text"].endswith(" s") == arm["finished"]
                if not arm["finished"]:
                    assert arm["w"] == "0.000" and not arm["wins"]


def test_readout_claim_names_the_tall_cell_and_the_rivals():
    claim = home_readout.readout()["claim"]
    assert claim.startswith("Tall scenario, same pod, 16M x 128: on one GPU bonsai fits in ")
    assert "XGBoost, LightGBM and CatBoost take " in claim


def test_door_counts_read_their_sources():
    counts = home_readout.door_counts(NAV)
    assert counts == {"learn": 3, "api": 44, "results": len(render_results.PERF_AXES)}


def test_the_live_nav_holds_every_chapter():
    assert home_readout.door_counts(_live_nav())["learn"] == 27


def _page(src_uri: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(meta={}, file=types.SimpleNamespace(src_uri=src_uri))


def test_count_span_is_filled_only_on_the_landing_page():
    markdown = '<span class="bz-count" data-bz-count="api"></span>\n**[API](api/install.md)**'
    landing = _page("index.md")
    out = home_readout.on_page_markdown(markdown, landing, {"nav": NAV}, None)
    assert '<span class="bz-count" data-bz-count="api">44<small>parameters</small></span>' in out
    assert "readout" in landing.meta
    chapter = _page("guide/README.md")
    assert home_readout.on_page_markdown(markdown, chapter, {"nav": NAV}, None) == markdown
    assert chapter.meta == {}


def test_missing_row_raises_instead_of_rendering_empty(monkeypatch):
    rows = render_results.summary_rows()
    first = rows[0]
    absent = render_results.SummaryCell("-", "-", None, None, False, False)
    cells = {**first.cells, render_results.PLANE_GPU: absent}
    broken = [render_results.SummaryRow(first.grower, first.rival, cells), *rows[1:]]
    monkeypatch.setattr(render_results, "summary_rows", lambda axes=None: broken)
    with pytest.raises(home_readout.ReadoutError):
        home_readout.readout()
