"""Tests for the local half of scripts/standings_refresh.py.

The parity gate, the A/B verdict, and the supersession that turns a results
directory into a branch. The pod-driving half is not here: it rents hardware.

    pytest python/tests/bench/test_standings_refresh.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import standings_refresh  # noqa: E402


def _jsonl(path: pathlib.Path, *rows: dict) -> pathlib.Path:
    """Write rows as jsonl and return the path."""
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _parity_row(arm: str, fit_s: float, peak_rss_gb: float,
                **extra) -> dict:
    """One parity row at the anchor cell."""
    return {"rows": 4000000, "cols": 512, "grower": "cuda_depthwise",
            "arm": arm, "fit_s": fit_s, "peak_rss_gb": peak_rss_gb, **extra}


# Parity ===========================================================================================

def test_parity_absence_fails_unless_the_operator_accepts_it(tmp_path):
    """A missing file means the check never ran, which is the failure the gate
    exists to catch, so only --no-parity may read it as a pass."""
    missing = tmp_path / "parity.jsonl"

    text, ok = standings_refresh._parity(missing)
    assert not ok
    assert text == ("Parity check FAILED: no parity.jsonl in this results "
                    "dir. Absence is not evidence the check does not apply, "
                    "it means the check never ran.")

    text, ok = standings_refresh._parity(missing, allow_absent=True)
    assert ok
    assert text == ("Parity check absent (no parity.jsonl in this results "
                    "dir).")


def test_parity_passes_when_the_host_had_no_device_to_check_on(tmp_path):
    """A file of nothing but skips is the check declaring itself
    inapplicable, which is a pass with a caveat rather than a gap."""
    path = _jsonl(tmp_path / "parity.jsonl", {"skipped": True},
                  {"skipped": True})

    assert standings_refresh._parity(path) == (
        "Parity check skipped on this host (no visible CUDA device).", True)


def test_parity_tables_both_metrics_and_the_two_step_split(tmp_path):
    """The whole passing table: a median per arm per metric, a signed delta,
    the ingest/train split the perf page publishes, and the verdict."""
    path = _jsonl(
        tmp_path / "parity.jsonl",
        _parity_row("fused", 10.0, 4.0),
        _parity_row("fused", 12.0, 4.0),
        _parity_row("two_step", 10.2, 4.1, ingest_s=2.0, train_s=8.2),
        _parity_row("two_step", 11.0, 4.1, ingest_s=3.0, train_s=8.0))

    text, ok = standings_refresh._parity(path)
    assert ok
    assert text == (
        "| metric (4000000x512 cuda_depthwise) | fused | two-step | delta |\n"
        "|---|--:|--:|--:|\n"
        "| fit_s | 11.00s | 10.60s | -3.6% |\n"
        "| peak_rss_gb | 4.00GB | 4.10GB | +2.5% |\n"
        "\n"
        "Two-step split: ingest 2.50s, train 8.10s.\n"
        "\n"
        "Verdict: PASS (band +-5%).")


def test_parity_fails_the_gate_on_either_metric(tmp_path):
    """Peak RSS is a standings claim too: a two-step form that lost its
    device hint bins on the host, which costs memory before it costs time."""
    time_moved = _jsonl(
        tmp_path / "time.jsonl",
        _parity_row("fused", 10.0, 4.0), _parity_row("two_step", 14.0, 4.0))
    text, ok = standings_refresh._parity(time_moved)
    assert not ok
    assert "| fit_s | 10.00s | 14.00s | +40.0% **FAIL** |" in text
    assert "Verdict: FAIL (band +-5%)." in text

    rss_moved = _jsonl(
        tmp_path / "rss.jsonl",
        _parity_row("fused", 10.0, 4.0), _parity_row("two_step", 10.0, 9.0))
    text, ok = standings_refresh._parity(rss_moved)
    assert not ok
    assert "| peak_rss_gb | 4.00GB | 9.00GB | +125.0% **FAIL** |" in text


def test_parity_reports_a_missing_arm_as_not_applicable(tmp_path):
    """One arm alone is not a comparison. The row says so and the verdict
    stays a pass, because nothing was measured to disagree."""
    path = _jsonl(tmp_path / "parity.jsonl",
                  _parity_row("fused", 10.0, 4.0),
                  _parity_row("fused", 11.0, 4.2))

    text, ok = standings_refresh._parity(path)
    assert ok
    assert "| fit_s | n/a | n/a | n/a |" in text
    assert "| peak_rss_gb | n/a | n/a | n/a |" in text
    assert "Two-step split" not in text
    assert "Verdict: PASS (band +-5%)." in text


def test_parity_skips_the_split_when_the_rows_carry_no_ingest(tmp_path):
    """The split line is only printed when the two-step arm reported one."""
    path = _jsonl(tmp_path / "parity.jsonl",
                  _parity_row("fused", 10.0, 4.0),
                  _parity_row("two_step", 10.1, 4.0))

    text, ok = standings_refresh._parity(path)
    assert ok
    assert "Two-step split" not in text


# Verdict ==========================================================================================

def _ab_row(arm: str, fit_s, peak_rss_gb, rows: int = 4000000) -> dict:
    """One A/B row."""
    return {"rows": rows, "cols": 512, "grower": "cuda_depthwise", "arm": arm,
            "fit_s": fit_s, "peak_rss_gb": peak_rss_gb}


def test_verdict_is_empty_without_an_ab_file(tmp_path):
    """An A/B is optional (measure --prev-version drives it), so its absence
    is silence rather than a failure."""
    assert standings_refresh._verdict(tmp_path / "ab.jsonl") == ""


def test_verdict_tables_every_cell_and_flags_the_ones_that_moved(tmp_path):
    """One row per cell, both metrics, and the moved flag on the row whose
    delta left the 5% band. The cells sort, so the table is stable."""
    path = _jsonl(
        tmp_path / "ab.jsonl",
        _ab_row("old", 10.0, 4.0), _ab_row("new", 10.2, 4.1),
        _ab_row("old", 20.0, 8.0, rows=8000000),
        _ab_row("new", 14.0, 8.1, rows=8000000))

    assert standings_refresh._verdict(path) == (
        "| cell | grower | old | new | delta | old RSS | new RSS | "
        "RSS delta |\n"
        "|---|---|--:|--:|--:|--:|--:|--:|\n"
        "| 4000000x512 | cuda_depthwise | 10.00s | 10.20s | +2.0% | 4.00GB "
        "| 4.10GB | +2.5% |\n"
        "| 8000000x512 | cuda_depthwise | 20.00s | 14.00s | -30.0% | 8.00GB "
        "| 8.10GB | +1.2% **moved** |")


def test_verdict_moves_on_memory_alone(tmp_path):
    """Memory is a standings claim: a path change can cost RSS before it
    costs seconds, and that still ends the automatic merge."""
    path = _jsonl(tmp_path / "ab.jsonl",
                  _ab_row("old", 10.0, 4.0), _ab_row("new", 10.1, 6.0))

    assert standings_refresh._verdict(path).endswith(
        "| 4.00GB | 6.00GB | +50.0% **moved** |")


def test_verdict_reports_a_cell_with_one_arm_as_not_applicable(tmp_path):
    """A cell the old wheel could not run has nothing to compare against, and
    a metric no row reported is the same case one column over."""
    path = _jsonl(tmp_path / "ab.jsonl",
                  _ab_row("new", 10.0, 4.0),
                  _ab_row("old", 20.0, None, rows=8000000),
                  _ab_row("new", 20.4, None, rows=8000000))

    lines = standings_refresh._verdict(path).splitlines()
    assert lines[2] == ("| 4000000x512 | cuda_depthwise | n/a | n/a | n/a | "
                        "n/a | n/a | n/a |")
    assert lines[3] == ("| 8000000x512 | cuda_depthwise | 20.00s | 20.40s | "
                        "+2.0% | n/a | n/a | n/a |")


# Supersession =====================================================================================

STUB_UPDATE = """import json, pathlib, sys
log = pathlib.Path(__file__).resolve().parents[1] / "calls.jsonl"
with log.open("a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
"""

STUB_RENDER = """import pathlib
out = pathlib.Path(__file__).resolve().parents[1] / "docs/method/results.md"
out.write_text("rendered\\n")
"""


def _git(repo: pathlib.Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, check=True)
    return done.stdout.strip()


def _fake_repo(monkeypatch, tmp_path) -> pathlib.Path:
    """A committed repo whose scripts/ holds stand-ins for the real children.

    supersede runs update_standings.py and render_results.py as subprocesses
    and then commits with git, so the pin needs a real repo and real child
    processes. Only what those two children do is stood in for: the stub
    update_standings records the command line it was given, which is the
    contract supersede is actually responsible for.
    """
    repo = tmp_path / "repo"
    (repo / "benchmarks" / "results").mkdir(parents=True)
    (repo / "docs" / "method").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "README.md").write_text("readme\n")
    (repo / "benchmarks" / "results" / ".keep").write_text("")
    (repo / "docs" / "method" / "results.md").write_text("old\n")
    (repo / "scripts" / "update_standings.py").write_text(STUB_UPDATE)
    (repo / "scripts" / "render_results.py").write_text(STUB_RENDER)
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "refresh@test")
    _git(repo, "config", "user.name", "refresh")
    _git(repo, "add", "README.md", "benchmarks", "docs", "scripts")
    _git(repo, "commit", "-qm", "base")
    monkeypatch.setattr(standings_refresh, "REPO", repo)
    monkeypatch.setattr(standings_refresh, "RESULTS",
                        repo / "benchmarks" / "results")
    return repo


def _args(src: pathlib.Path, axes: str, **over) -> argparse.Namespace:
    """A supersede argument namespace."""
    return argparse.Namespace(results_dir=str(src), axes=axes, no_pr=True,
                              no_parity=False, **over)


def _session(tmp_path, *, parity: bool = True) -> pathlib.Path:
    """A pulled results directory: two axis files and the parity rows."""
    src = tmp_path / "session"
    src.mkdir()
    _jsonl(src / "gpu-tall-2026-09.jsonl",
           {"host": {"name": "pod-blackwell"}, "git_sha": "abc1234"})
    _jsonl(src / "cpu-tall-2026-09.jsonl",
           {"host": {"name": "pod-blackwell"}, "git_sha": "abc1234"})
    if parity:
        _jsonl(src / "parity.jsonl",
               _parity_row("fused", 10.0, 4.0),
               _parity_row("two_step", 10.1, 4.0))
    return src


def test_supersede_refuses_a_results_dir_with_no_parity_evidence(monkeypatch,
                                                                 tmp_path,
                                                                 capsys):
    """The anchor axis is in this session, so its parity rows should be here.
    Absence stops the supersession before anything is copied or committed."""
    _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path, parity=False)

    assert standings_refresh.supersede(_args(src, "gpu-tall")) == 1
    err = capsys.readouterr().err
    assert "no parity.jsonl in this results dir" in err
    assert not (standings_refresh.RESULTS / "gpu-tall-2026-09.jsonl").exists()


def test_supersede_refuses_a_failed_parity_band(monkeypatch, tmp_path, capsys):
    """A fused/two-step disagreement means the published ingest/train split
    describes a pipeline no cuda grower runs, so it must not ship."""
    _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path)
    _jsonl(src / "parity.jsonl", _parity_row("fused", 10.0, 4.0),
           _parity_row("two_step", 20.0, 4.0))

    assert standings_refresh.supersede(_args(src, "gpu-tall")) == 1
    assert "fused/two-step parity failed" in capsys.readouterr().err


def test_supersede_accepts_a_session_that_anchors_no_parity(monkeypatch,
                                                            tmp_path, capsys):
    """The pod takes parity rows only for the session measuring gpu-tall, so
    for any other session absence is the expected state, said out loud."""
    repo = _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path, parity=False)

    assert standings_refresh.supersede(_args(src, "cpu-tall")) == 0
    out = capsys.readouterr().out
    assert "Parity not expected" in out
    assert "Parity check absent" in out
    assert json.loads((repo / "calls.jsonl").read_text().strip()) == [
        "--axis", "cpu-tall", "--file", "cpu-tall-2026-09.jsonl"]


def test_supersede_refuses_a_results_dir_missing_an_axis(monkeypatch, tmp_path,
                                                         capsys):
    """Every requested axis must have delivered a file; a silently short
    session is the drift the axis list exists to catch."""
    _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path)

    assert standings_refresh.supersede(
        _args(src, "gpu-tall,gpu-wide")) == 1
    assert "no gpu-wide-*.jsonl in" in capsys.readouterr().err


def test_supersede_commits_the_refresh_on_a_branch(monkeypatch, tmp_path,
                                                   capsys):
    """The whole --no-pr path: the newest file per axis is copied in, the
    parity rows are committed as the anchor axis's companion and dated from
    it, update_standings is driven once per axis with the companion only on
    the anchor, and the commit names the axes and the machine the cpu ones
    were measured on."""
    repo = _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path)
    _jsonl(src / "gpu-tall-2026-08.jsonl",
           {"host": {"name": "pod-blackwell"}, "git_sha": "9999999"})

    assert standings_refresh.supersede(
        _args(src, "gpu-tall,cpu-tall")) == 0

    results = standings_refresh.RESULTS
    assert not (results / "gpu-tall-2026-08.jsonl").exists()
    assert (results / "cpu-tall-2026-09.jsonl").exists()
    assert (results / "parity-2026-09.jsonl").read_text() == (
        (src / "parity.jsonl").read_text())
    assert [json.loads(ln) for ln in
            (repo / "calls.jsonl").read_text().splitlines()] == [
        ["--axis", "gpu-tall", "--file", "gpu-tall-2026-09.jsonl",
         "--companion", "parity-2026-09.jsonl"],
        ["--axis", "cpu-tall", "--file", "cpu-tall-2026-09.jsonl"]]

    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").startswith(
        "standings-refresh-")
    message = _git(repo, "log", "-1", "--pretty=%B")
    assert message.startswith("bench(standings): refresh gpu-tall,cpu-tall")
    assert "Axes: gpu-tall,cpu-tall." in message
    assert "CPU axes (cpu-tall) were measured on pod-blackwell" in message
    committed = _git(repo, "show", "--name-only", "--pretty=", "HEAD")
    assert "benchmarks/results/parity-2026-09.jsonl" in committed
    assert "docs/method/results.md" in committed
    assert "committed on standings-refresh-" in capsys.readouterr().out


def test_supersede_titles_a_wide_refresh_by_count(monkeypatch, tmp_path):
    """The commit hook hard-caps titles at 72 characters, which a full axis
    list exceeds, so past the 50-character target the title carries the
    count and the body carries the list."""
    repo = _fake_repo(monkeypatch, tmp_path)
    src = tmp_path / "session"
    src.mkdir()
    axes = ["gpu-tall", "gpu-wide", "gpu-extreme", "cpu-tall", "cpu-wide"]
    for axis in axes:
        _jsonl(src / f"{axis}-2026-09.jsonl",
               {"host": {"name": "pod-blackwell"}, "git_sha": "abc1234"})
    _jsonl(src / "parity.jsonl", _parity_row("fused", 10.0, 4.0),
           _parity_row("two_step", 10.1, 4.0))

    assert standings_refresh.supersede(_args(src, ",".join(axes))) == 0

    assert _git(repo, "log", "-1", "--pretty=%s") == (
        "bench(standings): refresh 5 axes")


def test_supersede_notes_no_cpu_host_when_no_cpu_axis_ran(monkeypatch,
                                                          tmp_path):
    """The hosts note exists to record which ceiling stands behind a cpu
    number, so a gpu-only refresh must not carry one."""
    repo = _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path)

    assert standings_refresh.supersede(_args(src, "gpu-tall")) == 0

    assert "CPU axes" not in _git(repo, "log", "-1", "--pretty=%B")
