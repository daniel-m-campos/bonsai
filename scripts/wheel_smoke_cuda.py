"""CUDA-wheel contract on a GPU host: the release gate's on-pod smoke.

Run inside the python the wheel was installed into, on a machine with an
NVIDIA driver. Mirrors the [cuda] test suite's contracts at wheel level:
both cuda growers train, predictions track their CPU twins, and models
round-trip through save/load byte-stably. The GPU/CPU tolerance bands are
pre-registered here, not tuned after a failure: per-tree leaf values agree
to 1e-4 (tests/unit/test_cuda_grower.cpp), so 30 shrunk trees get 1e-3 on
predictions and half a point of r2.
"""

import os
import tempfile

import bonsai
import numpy as np

assert bonsai.cuda_available() is True, "no CUDA device visible to the wheel"

rng = np.random.default_rng(0)
X = rng.random((20_000, 16), dtype=np.float32)
y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.normal(0, 0.1, len(X))).astype(np.float32)

PRED_TOL = 1e-3
R2_TOL = 0.005

for gpu, cpu in (("cuda_depthwise", "depthwise"), ("cuda_oblivious", "oblivious")):
    mg = bonsai.BonsaiRegressor(n_iters=30, grower=gpu).fit(X, y)
    mc = bonsai.BonsaiRegressor(n_iters=30, grower=cpu).fit(X, y)
    pg = mg.predict(X)
    gap = float(np.max(np.abs(pg - mc.predict(X))))
    r2g, r2c = mg.score(X, y), mc.score(X, y)
    print(f"{gpu}: r2 {r2g:.4f} (cpu twin {r2c:.4f}), max |gpu-cpu| pred gap {gap:.2e}")
    assert gap < PRED_TOL, f"{gpu} prediction gap {gap} exceeds {PRED_TOL}"
    assert abs(r2g - r2c) < R2_TOL, f"{gpu} r2 {r2g} off cpu {r2c}"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.msgpack")
        mg.save(path)
        restored = bonsai.BonsaiRegressor.from_file(path)
        assert np.array_equal(restored.predict(X), pg), f"{gpu} save/load drifted"

print("cuda smoke OK")
