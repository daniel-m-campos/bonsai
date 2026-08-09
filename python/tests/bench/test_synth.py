"""Tests for bonsai.bench.synth (the byte-stability contract)."""

from __future__ import annotations

import hashlib

from bonsai.bench import synth

# Captured from the pre-move scripts/bench_scaling.py::gen_data, before
# bonsai.bench.synth (2026-07-14): the generator must stay byte-stable or
# every perf-division result loses comparability.
GEN_DATA_GOLDENS = {
    (10_000, 20, 42, 1000, 20): ["33172850a22efea3", "72dd8c4d554f8e15",
                                 "2a69346b81ee31cb", "699d900c1abda049"],
    (5000, 7, 0, 500, 5): ["4052a735cd2e4edf", "bf4a1f3766b784c6",
                           "62e923970a54e7a9", "2e46ac59454ce2d1"],
}


def test_gen_data_bytestable():
    for args, want in GEN_DATA_GOLDENS.items():
        got = [hashlib.sha256(a.tobytes()).hexdigest()[:16]
               for a in synth.gen_data(*args)]
        assert got == want, (args, got, want)
