"""Tests for scripts/check_standings.py's per-plane freshness gates."""

from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_standings  # noqa: E402


def _fake_tree(root: pathlib.Path) -> None:
    """A miniature repo: one host source, one device source, one harness."""
    (root / "src" / "cuda").mkdir(parents=True)
    (root / "include" / "bonsai" / "cuda").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "src" / "tree.cpp").write_text("host\n")
    (root / "src" / "cuda" / "kernel.cu").write_text("device\n")
    (root / "include" / "bonsai" / "tree.hpp").write_text("host header\n")
    (root / "include" / "bonsai" / "cuda" / "plane.hpp").write_text("device h\n")
    (root / "scripts" / "model_hash.py").write_text("harness\n")


def _point_at(monkeypatch, root: pathlib.Path) -> None:
    monkeypatch.setattr(check_standings, "REPO", root)
    monkeypatch.setattr(check_standings, "MODEL_HASH_SCRIPT",
                        root / "scripts" / "model_hash.py")


def test_cuda_only_change_leaves_the_cpu_plane_current(monkeypatch, tmp_path):
    """The whole point of registry v2: a device-only edit must not force a
    CPU-plane re-measurement, while the gpu axes go stale immediately."""
    _fake_tree(tmp_path)
    _point_at(monkeypatch, tmp_path)
    cpu_before = check_standings.plane_digest(check_standings.PLANE_CPU)
    gpu_before = check_standings.plane_digest(check_standings.PLANE_GPU)
    assert cpu_before != gpu_before

    reg = {"cpu-tall": {"sha": "abc1234", "plane": "cpu",
                        "hash_set": cpu_before, "refreshed_for": "1.7.0"},
           "gpu-tall": {"sha": "abc1234", "plane": "gpu",
                        "hash_set": gpu_before, "refreshed_for": "1.7.0"}}
    assert check_standings.stale_axes(reg) == []

    (tmp_path / "src" / "cuda" / "kernel.cu").write_text("device, faster\n")
    assert check_standings.plane_digest(check_standings.PLANE_CPU) == cpu_before
    assert check_standings.plane_digest(check_standings.PLANE_GPU) != gpu_before
    assert check_standings.stale_axes(reg) == ["gpu-tall"]
    assert check_standings.hash_skip("cpu-tall", reg["cpu-tall"])[0]
    ok, reason = check_standings.hash_skip("gpu-tall", reg["gpu-tall"])
    assert not ok and "gpu-plane sources changed" in reason

    # A host-side edit moves both planes: nothing proves a cpu axis current.
    (tmp_path / "src" / "tree.cpp").write_text("host, faster\n")
    assert check_standings.plane_digest(check_standings.PLANE_CPU) != cpu_before
    assert check_standings.stale_axes(reg) == ["cpu-tall", "gpu-tall"]


def test_release_gate_names_the_never_refreshed_axis(monkeypatch, tmp_path):
    """A registry v2 placeholder (file and sha null) must fail loudly rather
    than pass as "nothing changed"."""
    _fake_tree(tmp_path)
    _point_at(monkeypatch, tmp_path)
    reg = {"gpu-wide": {"file": None, "sha": None, "plane": "gpu",
                        "refreshed_for": None}}
    errors = check_standings.check_release(reg, "1.7.0")
    assert len(errors) == 1 and "never refreshed" in errors[0]
    assert check_standings.stale_axes(reg) == ["gpu-wide"]


def test_quality_axis_keeps_the_whole_implementation_digest(monkeypatch,
                                                            tmp_path):
    """Quality axes carry no plane and are gated on the full digest, which is
    the gpu plane's by construction, so decision 92's rule is unchanged."""
    _fake_tree(tmp_path)
    _point_at(monkeypatch, tmp_path)
    assert (check_standings.plane_digest(check_standings.PLANE_GPU)
            == check_standings.plane_digest(check_standings.PLANE_GPU))
    entry = {"sha": "abc1234", "hash_set": check_standings.plane_digest(check_standings.PLANE_GPU)}
    assert check_standings.hash_skip("quality-grinsztajn", entry)[0]
    (tmp_path / "src" / "cuda" / "kernel.cu").write_text("device, faster\n")
    assert not check_standings.hash_skip("quality-grinsztajn", entry)[0]


# The moved-verdict gate ===========================================================================

DECISIONS = """## 120. Something else

Standings: cpu-tall

## 121. The gpu-tall move ships

The new build trades 3% of fit time for the levelwise device planes.

Standings: gpu-tall

Cited: ab-gpu-2026-09.jsonl.
"""


def _ab_registry(monkeypatch, tmp_path, rows: list[dict],
                 decisions: str = DECISIONS) -> dict:
    """A registry whose gpu-tall carries an ab file holding ``rows``."""
    results = tmp_path / "benchmarks" / "results"
    results.mkdir(parents=True)
    (results / "ab-gpu-2026-09.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "decisions.md").write_text(decisions)
    monkeypatch.setattr(check_standings, "RESULTS", results)
    monkeypatch.setattr(check_standings, "DECISIONS", tmp_path / "decisions.md")
    return {"gpu-tall": {"plane": "gpu", "as_of_decision": 121,
                         "ab": "ab-gpu-2026-09.jsonl"},
            "cpu-tall": {"plane": "cpu", "as_of_decision": 120}}


def _arm(arm: str, fit_s: float, rss: float = 4.0, **extra) -> dict:
    return {"rows": 4000000, "cols": 512, "grower": "cuda_depthwise",
            "arm": arm, "fit_s": fit_s, "peak_rss_gb": rss, **extra}


def test_a_held_ab_needs_no_citation(monkeypatch, tmp_path):
    reg = _ab_registry(monkeypatch, tmp_path,
                       [_arm("anchor", 10.0), _arm("old", 10.1), _arm("new", 10.1)],
                       decisions="## 1. Nothing\n")
    assert check_standings.check_ab(reg) == []


def test_an_axis_without_an_ab_file_is_not_gated(monkeypatch, tmp_path):
    reg = _ab_registry(monkeypatch, tmp_path, [], decisions="## 1. Nothing\n")
    del reg["gpu-tall"]["ab"]
    assert check_standings.check_ab(reg) == []
    reg["gpu-tall"]["ab"] = "ab-gpu-2026-10.jsonl"
    assert check_standings.check_ab(reg) == []


def test_a_moved_ab_nobody_cites_fails_the_gate(monkeypatch, tmp_path):
    reg = _ab_registry(monkeypatch, tmp_path,
                       [_arm("anchor", 10.0), _arm("old", 10.0), _arm("new", 10.3)],
                       decisions="## 1. Nothing\n")
    errors = check_standings.check_ab(reg)
    assert len(errors) == 1
    assert errors[0].startswith(
        "gpu-tall: the release A/B ab-gpu-2026-09.jsonl moved "
        "(4000000x512 cuda_depthwise: fit vs the 1.15.0 anchor +3.0% "
        "(band 2%)); a decision entry tagged `Standings: gpu-tall` must cite "
        "ab-gpu-2026-09.jsonl")


def test_a_moved_ab_ships_under_an_entry_tagged_with_its_axis(monkeypatch,
                                                              tmp_path):
    reg = _ab_registry(monkeypatch, tmp_path,
                       [_arm("anchor", 10.0), _arm("old", 10.0), _arm("new", 10.3)])
    assert check_standings.check_ab(reg) == []
    assert check_standings.check_decisions(reg) == []


def test_a_citation_under_another_axis_tag_does_not_count(monkeypatch,
                                                          tmp_path):
    reg = _ab_registry(monkeypatch, tmp_path,
                       [_arm("anchor", 10.0), _arm("old", 10.0), _arm("new", 10.3)],
                       decisions=DECISIONS.replace("Standings: gpu-tall",
                                                   "Standings: cpu-tall"))
    errors = check_standings.check_ab(reg)
    assert len(errors) == 1 and errors[0].startswith("gpu-tall: the release A/B")


def test_a_skipped_arm_carries_no_cell(monkeypatch, tmp_path):
    reg = _ab_registry(monkeypatch, tmp_path,
                       [{"arm": "anchor", "grower": "cuda_depthwise",
                         "skipped": "no cuda"},
                        _arm("old", 10.0), _arm("new", 10.1)],
                       decisions="## 1. Nothing\n")
    assert check_standings.check_ab(reg) == []
    rows = check_standings.ab_rows(tmp_path / "benchmarks" / "results"
                                   / "ab-gpu-2026-09.jsonl")
    assert [r["arm"] for r in rows] == ["old", "new"]


def test_tagged_entries_carry_every_axis_of_a_tag_and_skip_untagged_ones():
    text = ("## 7. Untagged\n\nbody\n\n## 8. Two axes\n\n"
            "Standings: gpu-tall, cpu-tall\n\n## 9. One\n\nStandings: cpu-wide\n")
    assert [(n, axes) for n, axes, _ in check_standings.tagged_entries(text)] == [
        (8, ["gpu-tall", "cpu-tall"]), (9, ["cpu-wide"])]
    assert set(check_standings.tagged_bodies(text)) == {
        "gpu-tall", "cpu-tall", "cpu-wide"}


def test_check_decisions_still_holds_a_tag_newer_than_the_registry(monkeypatch,
                                                                    tmp_path):
    reg = _ab_registry(monkeypatch, tmp_path, [])
    reg["gpu-tall"]["as_of_decision"] = 120
    errors = check_standings.check_decisions(reg)
    assert errors == [
        "decision 121 supersedes the 'gpu-tall' standings (as_of_decision "
        "120): refresh the axis and bump the registry, or drop the tag if "
        "the claim does not move the standings"]
    reg["gpu-tall"]["as_of_decision"] = 121
    assert check_standings.check_decisions({"gpu-tall": reg["gpu-tall"]}) == [
        "decision 120: unknown standings axis 'cpu-tall' (known: ['gpu-tall'])"]
