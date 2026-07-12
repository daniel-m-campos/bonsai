# The byte-identity gate: trains a fixed 500k x 100 depthwise model and
# prints a short sha256 of the saved msgpack. Any refactor that claims to
# preserve CPU behavior must leave this hash unchanged; capture the baseline
# BEFORE touching anything (see .claude/skills/quality-gates). Fixed seed,
# fixed threads — the determinism contract (decision 49) makes the hash
# stable across runs on one machine.
#
#   make python && PYTHONPATH=build/python python3 scripts/model_hash.py
import hashlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, "build/python")
import bonsai

rng = np.random.default_rng(np.random.SeedSequence([42, 500_000, 100]))
X = rng.random((500_000, 100), dtype=np.float32)
y = (X[:, :20].reshape(-1, 4, 5) * (0.6 ** np.arange(4))[None, :, None]) \
    .sum(axis=(1, 2)).astype(np.float32)
y += rng.normal(0, y.std() * 0.33, len(y)).astype(np.float32)

def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# Printed so a cross-platform hash mismatch can be attributed: if the DATA
# digests differ, numpy built different inputs (SIMD-width-dependent
# reduction trees); only if they match is the divergence bonsai's.
print("data:", _sha(X), _sha(y))

pairs = [("dispatch.grower_name", "depthwise"), ("booster.n_iters", "20"),
         ("booster.learning_rate", "0.1"), ("tree.max_depth", "8"),
         ("bin_mapper.max_bin", "255"), ("parallel.n_threads", "8")]


def _model_sha(extra=()) -> str:
    m = bonsai.train([*pairs, *extra], X, y)
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as f:
        m.save(f.name)
        return hashlib.sha256(open(f.name, "rb").read()).hexdigest()[:16]


# The full-column variant skips the mapper's subsampling entirely: if it
# matches across platforms while the default diverges, the divergence
# lives in the sampling path (std::ranges::sample is implementation-
# defined), not in training arithmetic.
print("fullsample:", _model_sha([("bin_mapper.n_samples", "500000")]))
print("sha256:", _model_sha())
