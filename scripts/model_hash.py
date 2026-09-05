"""The byte-identity gate: trains a fixed 500k x 100 depthwise model and
prints a short sha256 of the saved msgpack. Any refactor that claims to
preserve CPU behavior must leave this hash unchanged; capture the baseline
BEFORE touching anything (see .claude/skills/quality-gates). Fixed seed,
fixed threads: the determinism contract (decision 49) makes the hash
stable across runs on one machine.

    make python && PYTHONPATH=build/python python3 scripts/model_hash.py

`--grower cuda_depthwise` measures a device plane instead (needs the CUDA
build on PYTHONPATH); the device contract is identity run to run on one
device and build, so the gate there is three runs printing one hash.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, "build/python")
import bonsai

PAIRS = {"dispatch.grower_name": "depthwise", "booster.n_iters": "20",
         "booster.learning_rate": "0.1", "tree.max_depth": "8",
         "bin_mapper.max_bin": "255", "parallel.n_threads": "8"}


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _gen_data() -> tuple[np.ndarray, np.ndarray]:
    """The fixed 500k x 100 input; the seed sequence is part of the gate."""
    rng = np.random.default_rng(np.random.SeedSequence([42, 500_000, 100]))
    X = rng.random((500_000, 100), dtype=np.float32)
    y = (X[:, :20].reshape(-1, 4, 5) * (0.6 ** np.arange(4))[None, :, None]) \
        .sum(axis=(1, 2)).astype(np.float32)
    y += rng.normal(0, y.std() * 0.33, len(y)).astype(np.float32)
    return X, y


def _model_sha(X: np.ndarray, y: np.ndarray, extra=()) -> str:
    m = bonsai.train({**PAIRS, **dict(extra)}, X, y)
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as f:
        m.save(f.name)
        return hashlib.sha256(open(f.name, "rb").read()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grower", default=PAIRS["dispatch.grower_name"])
    grower = parser.parse_args().grower
    PAIRS["dispatch.grower_name"] = grower
    X, y = _gen_data()
    # Printed so a cross-platform hash mismatch can be attributed: if the DATA
    # digests differ, numpy built different inputs (SIMD-width-dependent
    # reduction trees); only if they match is the divergence bonsai's.
    print("data:", _sha(X), _sha(y))
    # Attribution tiers for a cross-platform mismatch: full-sample kills the
    # mapper's sampling RNG; serial kills every parallel site; one iteration
    # kills accumulation drift. Whichever tier first diverges names the layer.
    # Two hashes, one contract at two thread counts: model bits depend only on
    # the input, the config, and the CONFIGURED thread count (the fill plan
    # scales block counts with it, per host-determinism), never on
    # the architecture (decisions 59/60). Both lines are asserted equal across
    # arm64/x86-64 by the cross-arch CI gate.
    print("serial_sha256:",
          _model_sha(X, y, {"bin_mapper.n_samples": "500000", "parallel.n_threads": "1"}))
    print("sha256:", _model_sha(X, y))
    return 0


if __name__ == "__main__":
    sys.exit(main())
