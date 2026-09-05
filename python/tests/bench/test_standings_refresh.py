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

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_standings  # noqa: E402
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

HEADER = ("| cell | grower | anchor | old | new | vs old | vs anchor "
          "| old RSS | new RSS | RSS delta |\n"
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")


def _ab_row(arm: str, fit_s, peak_rss_gb, rows: int = 4000000,
            version: str | None = None) -> dict:
    """One A/B row."""
    row = {"rows": rows, "cols": 512, "grower": "cuda_depthwise", "arm": arm,
           "fit_s": fit_s, "peak_rss_gb": peak_rss_gb}
    if version is not None:
        row["version"] = version
    return row


def _three_arms(anchor, old, new, rss=4.0, rows: int = 4000000) -> list[dict]:
    return [_ab_row("anchor", anchor, rss, rows), _ab_row("old", old, rss, rows),
            _ab_row("new", new, rss, rows)]


def test_verdict_is_empty_without_an_ab_file(tmp_path):
    """An A/B is optional (measure --prev-version drives it), so its absence
    is silence rather than a failure."""
    assert standings_refresh._verdict(tmp_path / "ab-gpu.jsonl") == ""


def test_verdict_tables_every_cell_and_flags_the_ones_that_moved(tmp_path):
    """One row per cell, three arms, both metrics, and the moved flag on the
    delta that left its band. The cells sort, so the table is stable."""
    path = _jsonl(
        tmp_path / "ab-gpu.jsonl",
        *_three_arms(10.1, 10.0, 10.2),
        _ab_row("anchor", 20.0, 8.0, rows=8000000),
        _ab_row("old", 20.0, 8.0, rows=8000000),
        _ab_row("new", 14.0, 8.1, rows=8000000))

    assert standings_refresh._verdict(path) == HEADER + (
        "| 4000000x512 | cuda_depthwise | 10.10s | 10.00s | 10.20s | +2.0% "
        "| +1.0% | 4.00GB | 4.00GB | +0.0% |\n"
        "| 8000000x512 | cuda_depthwise | 20.00s | 20.00s | 14.00s "
        "| -30.0% **moved** | -30.0% **moved** | 8.00GB | 8.10GB | +1.2% |")


def test_verdict_flags_a_cumulative_move_the_release_band_hides(tmp_path):
    """The case the anchor arm exists for: new sits inside the release band
    against old, and past the anchor band against the fixed wheel, because
    each release lost a little. Only the anchor column carries the flag."""
    path = _jsonl(tmp_path / "ab-cpu.jsonl", *_three_arms(100.0, 102.0, 104.0))

    assert standings_refresh._verdict(path).endswith(
        "| 100.00s | 102.00s | 104.00s | +2.0% | +4.0% **moved** "
        "| 4.00GB | 4.00GB | +0.0% |")
    assert check_standings.ab_moves(_three_arms(100.0, 102.0, 104.0)) == [
        f"4000000x512 cuda_depthwise: fit vs the "
        f"{check_standings.ANCHOR_VERSION} anchor +4.0% (band "
        f"{check_standings.ANCHOR_BAND_PCT}%)"]


def test_verdict_takes_the_min_over_reps_not_the_median(tmp_path):
    """Noise on a fixed workload only adds time, so one slow rep must not
    move a cell: the median of {10, 12, 12} would read +20%, the min reads
    the 10 both arms reached."""
    path = _jsonl(tmp_path / "ab-gpu.jsonl",
                  _ab_row("old", 10.0, 4.0), _ab_row("old", 10.0, 4.0),
                  _ab_row("old", 10.0, 4.0),
                  _ab_row("new", 10.0, 4.0), _ab_row("new", 12.0, 4.0),
                  _ab_row("new", 12.0, 4.0))

    assert standings_refresh._verdict(path).endswith(
        "| n/a | 10.00s | 10.00s | +0.0% | n/a | 4.00GB | 4.00GB | +0.0% |")


def test_verdict_flags_the_metric_that_moved_not_the_row(tmp_path):
    """A time move with no memory evidence lands on the time delta; the
    memory columns stay n/a rather than carrying a flag for a number they
    never had."""
    path = _jsonl(tmp_path / "ab-gpu.jsonl",
                  _ab_row("old", 20.0, None), _ab_row("new", 14.0, None))

    assert standings_refresh._verdict(path).endswith(
        "| n/a | 20.00s | 14.00s | -30.0% **moved** | n/a | n/a | n/a | n/a |")


@pytest.mark.parametrize("arm,band_name,column", [
    ("old", "AB_BAND_PCT", 5),
    ("anchor", "ANCHOR_BAND_PCT", 6),
])
def test_verdict_band_edge_is_the_named_band(tmp_path, arm, band_name, column):
    """Each comparison's flag fires past its own band, not at it, and the
    number the PR body quotes is the one the table applies."""
    band = getattr(standings_refresh, band_name)
    at = _jsonl(tmp_path / "at.jsonl",
                _ab_row(arm, 100.0, 4.0),
                _ab_row("new", 100.0 + band, 4.0))
    past = _jsonl(tmp_path / "past.jsonl",
                  _ab_row(arm, 100.0, 4.0),
                  _ab_row("new", 100.0 + band + 0.1, 4.0))

    assert "**moved**" not in standings_refresh._verdict(at)
    cells = standings_refresh._verdict(past).splitlines()[-1].split(" | ")
    assert cells[column] == f"+{band + 0.1:.1f}% **moved**"


def test_verdict_moves_on_memory_alone(tmp_path):
    """Memory is a standings claim: a path change can cost RSS before it
    costs seconds, and that still ends the automatic merge."""
    path = _jsonl(tmp_path / "ab-gpu.jsonl",
                  _ab_row("old", 10.0, 4.0), _ab_row("new", 10.1, 6.0))

    assert standings_refresh._verdict(path).endswith(
        "| 4.00GB | 6.00GB | +50.0% **moved** |")
    assert check_standings.ab_moves([_ab_row("old", 10.0, 4.0),
                                     _ab_row("new", 10.1, 6.0)]) == [
        "4000000x512 cuda_depthwise: RSS vs the previous release +50.0% "
        f"(band {check_standings.AB_BAND_PCT}%)"]


def test_verdict_reports_a_cell_with_one_arm_as_not_applicable(tmp_path):
    """A cell the wheels could not run has nothing to compare against, and
    a metric no row reported is the same case one column over."""
    path = _jsonl(tmp_path / "ab-gpu.jsonl",
                  _ab_row("new", 10.0, 4.0),
                  _ab_row("old", 20.0, None, rows=8000000),
                  _ab_row("new", 20.4, None, rows=8000000))

    lines = standings_refresh._verdict(path).splitlines()
    assert lines[2] == ("| 4000000x512 | cuda_depthwise | n/a | n/a | 10.00s "
                        "| n/a | n/a | n/a | 4.00GB | n/a |")
    assert lines[3] == ("| 8000000x512 | cuda_depthwise | n/a | 20.00s | 20.40s "
                        "| +2.0% | n/a | n/a | n/a | n/a |")


def test_verdicts_head_each_plane_with_the_wheels_it_fitted(tmp_path):
    """A results directory holds one A/B file per plane; the printed verdict
    names the plane, the file and the versions the arms resolved to, so a
    wheel that silently resolved to the wrong release is visible."""
    _jsonl(tmp_path / "ab-gpu.jsonl",
           _ab_row("anchor", 10.0, 4.0, version="1.15.0"),
           _ab_row("old", 10.0, 4.0, version="2.0.0"),
           _ab_row("new", 10.0, 4.0, version="2.1.0+source"))
    _jsonl(tmp_path / "ab-cpu.jsonl",
           _ab_row("old", 10.0, 4.0, version="2.0.0"),
           _ab_row("new", 10.0, 4.0, version="2.1.0+source"))

    text = standings_refresh._ab_verdicts(tmp_path)
    assert text.startswith(
        "gpu plane (ab-gpu.jsonl; anchor 1.15.0, old 2.0.0, new 2.1.0+source):"
        "\n\n" + HEADER)
    assert "\n\ncpu plane (ab-cpu.jsonl; old 2.0.0, new 2.1.0+source):\n\n" in text
    assert standings_refresh._ab_verdicts(tmp_path / "empty") == ""


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


def test_supersede_reads_axes_the_way_measure_does(monkeypatch, tmp_path):
    """A trailing comma or a space around a name is not an axis. measure
    already drops blanks; supersede reading the same flag differently made
    a valid session fail as missing an axis with no name."""
    repo = _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path, parity=False)

    assert standings_refresh.supersede(_args(src, " cpu-tall ,")) == 0
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


def test_supersede_commits_each_plane_ab_with_its_tall_axis(monkeypatch,
                                                            tmp_path):
    """The gpu A/B rides gpu-tall and the cpu A/B rides cpu-tall, each dated
    from the axis file it was measured beside, and update_standings learns
    of each through --ab so the file supersedes with the axis and the gate
    reads it from the registry."""
    repo = _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path)
    _jsonl(src / "ab-gpu.jsonl", *_three_arms(10.0, 10.0, 10.1))
    _jsonl(src / "ab-cpu.jsonl", *_three_arms(50.0, 50.0, 50.5))

    assert standings_refresh.supersede(
        _args(src, "gpu-tall,cpu-tall")) == 0

    results = standings_refresh.RESULTS
    assert (results / "ab-gpu-2026-09.jsonl").read_text() == (
        (src / "ab-gpu.jsonl").read_text())
    assert (results / "ab-cpu-2026-09.jsonl").read_text() == (
        (src / "ab-cpu.jsonl").read_text())
    assert [json.loads(ln) for ln in
            (repo / "calls.jsonl").read_text().splitlines()] == [
        ["--axis", "gpu-tall", "--file", "gpu-tall-2026-09.jsonl",
         "--companion", "parity-2026-09.jsonl", "--ab", "ab-gpu-2026-09.jsonl"],
        ["--axis", "cpu-tall", "--file", "cpu-tall-2026-09.jsonl",
         "--ab", "ab-cpu-2026-09.jsonl"]]
    committed = _git(repo, "show", "--name-only", "--pretty=", "HEAD")
    assert "benchmarks/results/ab-gpu-2026-09.jsonl" in committed
    assert "benchmarks/results/ab-cpu-2026-09.jsonl" in committed


def test_supersede_leaves_a_plane_ab_behind_when_its_axis_did_not_run(
        monkeypatch, tmp_path):
    """A session that measured only the gpu plane cannot date a cpu A/B: the
    file stays on the pod and the cpu-tall entry keeps whatever it carried."""
    repo = _fake_repo(monkeypatch, tmp_path)
    src = _session(tmp_path)
    _jsonl(src / "ab-gpu.jsonl", *_three_arms(10.0, 10.0, 10.1))
    _jsonl(src / "ab-cpu.jsonl", *_three_arms(50.0, 50.0, 50.5))

    assert standings_refresh.supersede(_args(src, "gpu-tall")) == 0

    results = standings_refresh.RESULTS
    assert (results / "ab-gpu-2026-09.jsonl").exists()
    assert not (results / "ab-cpu-2026-09.jsonl").exists()
    assert [json.loads(ln) for ln in
            (repo / "calls.jsonl").read_text().splitlines()] == [
        ["--axis", "gpu-tall", "--file", "gpu-tall-2026-09.jsonl",
         "--companion", "parity-2026-09.jsonl", "--ab", "ab-gpu-2026-09.jsonl"]]
