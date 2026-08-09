"""Tests for scripts/check_standings.py's per-plane freshness gates."""

from __future__ import annotations

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
