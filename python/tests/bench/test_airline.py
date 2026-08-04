"""Tests for bonsai.bench.airline (fetch's baked-path preference)."""

from __future__ import annotations

from bonsai.bench import airline


def test_fetch_prefers_baked_path_when_present(tmp_path, monkeypatch):
    baked = tmp_path / "baked"
    baked.mkdir()
    (baked / "train-10m.csv").write_text("baked-train")
    (baked / "test.csv").write_text("baked-test")
    monkeypatch.setattr(airline, "BAKED_DIR", baked)
    monkeypatch.setattr(airline, "data_root", lambda: tmp_path / "cache")

    def _fail_download(*args, **kwargs):
        raise AssertionError("fetch() must not reach S3 when baked files exist")

    monkeypatch.setattr(airline.urllib.request, "urlretrieve", _fail_download)

    train, test = airline.fetch("10m")

    assert train == baked / "train-10m.csv"
    assert test == baked / "test.csv"


def test_fetch_falls_back_to_download_when_baked_path_absent(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(airline, "BAKED_DIR", tmp_path / "no-such-dir")
    monkeypatch.setattr(airline, "data_root", lambda: cache)
    fetched = []

    def _fake_download(url, local):
        fetched.append(url)
        local.write_text("downloaded")

    monkeypatch.setattr(airline.urllib.request, "urlretrieve", _fake_download)

    train, test = airline.fetch("0.1m")

    assert train == cache / "train-0.1m.csv"
    assert test == cache / "airline_test.csv"
    assert fetched == [f"{airline.S3}/train-0.1m.csv", f"{airline.S3}/test.csv"]
