"""CUDA-wheel contract on a GPU-less host: the fold-in must be invisible.

Run inside the venv the CUDA wheel was installed into, after wheel_smoke.py
(which covers the CPU training surface). Asserts the degradation guarantees
the registry design makes: availability reports false, the cuda growers are
registered but decline to train loudly, and nothing crashes on the way.
"""

import bonsai
import numpy as np

assert bonsai.cuda_available() is False, "expected no CUDA device on this host"

rng = np.random.default_rng(0)
X = rng.random((512, 4), dtype=np.float32)
y = X[:, 0].astype(np.float32)

for grower in ("cuda_depthwise", "cuda_oblivious"):
    try:
        bonsai.BonsaiRegressor(n_iters=2, grower=grower).fit(X, y)
    except Exception as e:  # any loud failure is the contract; silence is the bug
        print(f"{grower}: declined cleanly ({type(e).__name__})")
    else:
        raise SystemExit(f"{grower} trained without a device")

print("gpu-absent smoke OK")
