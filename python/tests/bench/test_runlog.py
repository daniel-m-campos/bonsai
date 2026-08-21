"""Tests for bonsai.bench.runlog (schema round-trip)."""

from __future__ import annotations

import json
import tempfile

import pytest
from bonsai.bench import runlog


def test_runlog_roundtrip():
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as f:
        knobs = {"b": 2, "a": 1}
        row = runlog.emit_row(f.name, division="perf", suite="scaling",
                              knobs=knobs, timing_mode="in_memory",
                              variant="bonsai_dw", value=1.0, metric="r2",
                              status="ok")
        back = json.loads(open(f.name).read().splitlines()[-1])
        assert back == json.loads(json.dumps(row))
        for key in ("schema", "ts", "git_sha", "division", "suite", "cmd",
                    "timing_mode", "host", "knobs", "knobs_hash"):
            assert key in back, key
        # hash is canonical: key order must not matter
        assert runlog.knobs_hash({"a": 1, "b": 2}) == back["knobs_hash"]
    with pytest.raises(ValueError):
        runlog.emit_row("/tmp/x.jsonl", division="nope", suite="s")


def test_row_field_spellings_are_distinct():
    """Every Row constant names a different column: a collision would
    silently overwrite a row field on emit."""
    values = [v for k, v in vars(runlog.Row).items() if not k.startswith("_")]
    assert len(values) == len(set(values)), "duplicate Row field spelling"


def test_detect_host_driver_key():
    host = runlog.detect_host()
    assert "driver" in host
    assert host["driver"] is None or isinstance(host["driver"], str)
    assert "cpu_quota" in host
    assert "omp_wait_policy" in host


@pytest.mark.parametrize("text,expected", [("1360000 100000", 13.6),
                                           ("max 100000", None)])
def test_cpu_quota_reads_cgroup_v2(tmp_path, monkeypatch, text, expected):
    path = tmp_path / "cpu.max"
    path.write_text(text + "\n")
    monkeypatch.setattr(runlog, "CGROUP_V2_CPU_MAX", str(path))
    monkeypatch.setattr(runlog, "CGROUP_V1_QUOTA", str(tmp_path / "absent"))
    monkeypatch.setattr(runlog, "_cpuset_cores", lambda: None)
    assert runlog.cpu_quota() == expected


def test_cpu_quota_falls_back_to_cgroup_v1(tmp_path, monkeypatch):
    quota, period = tmp_path / "quota", tmp_path / "period"
    quota.write_text("2720000\n")
    period.write_text("100000\n")
    monkeypatch.setattr(runlog, "CGROUP_V2_CPU_MAX", str(tmp_path / "absent"))
    monkeypatch.setattr(runlog, "CGROUP_V1_QUOTA", str(quota))
    monkeypatch.setattr(runlog, "CGROUP_V1_PERIOD", str(period))
    monkeypatch.setattr(runlog, "_cpuset_cores", lambda: None)
    assert runlog.cpu_quota() == 27.2
    quota.write_text("-1\n")
    assert runlog.cpu_quota() is None


def test_cpu_quota_reads_the_cpuset_when_bandwidth_is_unlimited(monkeypatch):
    """A rented CPU pod caps by cpuset: cpu.max says max, 24 CPUs allowed."""
    monkeypatch.setattr(runlog, "_cfs_quota_cores", lambda: None)
    monkeypatch.setattr(runlog, "_cpuset_cores", lambda: 24.0)
    assert runlog.cpu_quota() == 24.0


def test_cpu_quota_takes_the_tighter_of_the_two_caps(monkeypatch):
    monkeypatch.setattr(runlog, "_cfs_quota_cores", lambda: 13.6)
    monkeypatch.setattr(runlog, "_cpuset_cores", lambda: 128.0)
    assert runlog.cpu_quota() == 13.6


def test_usable_ram_takes_the_cgroup_limit_over_the_host(tmp_path, monkeypatch):
    """The verified pod case: a 4GB container on a host reporting 1TB."""
    path = tmp_path / "memory.max"
    path.write_text(f"{4 * 2**30}\n")
    monkeypatch.setattr(runlog, "CGROUP_V2_MEMORY_MAX", str(path))
    monkeypatch.setattr(runlog, "_machine_ram_bytes", lambda: 1024 * 2**30)
    assert runlog.usable_ram_gb() == 4


@pytest.mark.parametrize("text", ["max", str(2**63 - 4096)])
def test_usable_ram_falls_back_to_the_host_when_uncapped(tmp_path, monkeypatch,
                                                         text):
    """`max` on v2 and the near-2**63 sentinel on v1 both mean no limit."""
    v2, v1 = tmp_path / "memory.max", tmp_path / "limit_in_bytes"
    (v2 if text == "max" else v1).write_text(text + "\n")
    monkeypatch.setattr(runlog, "CGROUP_V2_MEMORY_MAX", str(v2))
    monkeypatch.setattr(runlog, "CGROUP_V1_MEMORY_LIMIT", str(v1))
    monkeypatch.setattr(runlog, "_machine_ram_bytes", lambda: 64 * 2**30)
    assert runlog.usable_ram_gb() == 64


def test_usable_ram_reads_the_cgroup_v1_limit(tmp_path, monkeypatch):
    path = tmp_path / "limit_in_bytes"
    path.write_text(f"{8 * 2**30}\n")
    monkeypatch.setattr(runlog, "CGROUP_V2_MEMORY_MAX", str(tmp_path / "absent"))
    monkeypatch.setattr(runlog, "CGROUP_V1_MEMORY_LIMIT", str(path))
    monkeypatch.setattr(runlog, "_machine_ram_bytes", lambda: 1024 * 2**30)
    assert runlog.usable_ram_gb() == 8


def test_cpuset_cores_is_none_when_the_mask_is_the_whole_machine(monkeypatch):
    monkeypatch.setattr(runlog.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(runlog.os, "sched_getaffinity", lambda pid: set(range(8)),
                        raising=False)
    assert runlog._cpuset_cores() is None
    monkeypatch.setattr(runlog.os, "sched_getaffinity", lambda pid: {0, 1},
                        raising=False)
    assert runlog._cpuset_cores() == 2.0
