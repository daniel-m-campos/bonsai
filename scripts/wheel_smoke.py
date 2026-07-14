"""Wheel/sdist smoke: run inside the venv the artifact was installed into.

One script shared by the wheels build job and the sdist install test, so the
'every wheel is smoke-tested' guarantee cannot silently diverge between the
two paths. Covers the API surface that has actually broken before: fit,
predict_proba, Dataset reuse, and the save/from_file round-trip (a
tomllib-on-3.10 break shipped once because no smoke exercised it).
"""

import os
import tempfile

import bonsai
import numpy as np

rng = np.random.default_rng(0)
X = rng.random((2000, 8), dtype=np.float32)
y = (X[:, 0] * 2 + rng.normal(0, 0.1, 2000)).astype(np.float32)
r2 = bonsai.BonsaiRegressor(n_iters=20).fit(X, y).score(X, y)
assert r2 > 0.8, r2

yc = (X[:, 1] > 0.5).astype(np.float32)
clf = bonsai.BonsaiClassifier(n_iters=10).fit(X, yc)
assert clf.predict_proba(X).shape == (2000, 2)

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "clf.msgpack")
    clf.save(p)
    restored = bonsai.BonsaiClassifier.from_file(p)
    assert restored.predict(X).shape == (2000,)

ds = bonsai.Dataset(X, y)
bonsai.train([("booster.n_iters", "5")], ds)
print("smoke OK: r2 =", round(float(r2), 4))
