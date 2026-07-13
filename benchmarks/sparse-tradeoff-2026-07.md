# Sparse-input probe, 2026-07: the measured boundary

The last unmeasured candidate forcing function (crown-week plan): does sparse/high-dimensional input force users to a reference library? Method: rcv1 text (40k rows, CCAT target, top-k columns by document frequency), campaign-matched knobs; reproduce with `scripts/probe_sparse.py`.

## Results

| k cols | lgbm sparse | xgb sparse | lgbm densified | bonsai dense | exact EFB bundles |
|--:|--:|--:|--:|--:|--:|
| 1,000 | 0.9712 / 2.8s | 0.9694 / 5.3s | 0.9715 / 2.9s | 0.9712 / **20.8s** | 938 (1.07×) |
| 5,000 | 0.9716 / 4.8s | 0.9693 / 12.0s | 0.9710 / 6.4s | 0.9715 / **79.3s** | 1,258 (3.97×) |
| 10,000 | 0.9710 / 5.4s | 0.9693 / 14.3s | 0.9714 / 8.7s | 0.9715 / **137.0s** | — |

One-hot control (200k rows, 20 categoricals of cardinality 50): lightgbm on the 1,000-column one-hot expansion with its native EFB reaches 0.8647; bonsai on the **20 raw code columns** reaches **0.8708** — the EFB use-case collapses to "don't one-hot", which the encoder pipeline already does better.

## Verdict (feature-admission)

1. **Quality: no gap.** bonsai matches lightgbm-sparse to the third decimal at every width, and lightgbm's own sparse-vs-densified delta is noise — sparsity handling buys speed, not accuracy.
2. **Cost: a real forcing function above ~2k genuinely-sparse columns.** Dense histograms pay every zero: fit time scales linearly with width (25× at 10k tf-idf columns) and memory is O(rows × cols) — a 47k-column corpus at scale is simply out of reach dense.
3. **Preprocessing does NOT rescue text** — the one candidate this week's method failed on: exact conflict-free EFB bundling yields 1.07–4× column reduction against a ~600× sparsity factor. Bundling is a one-hot phenomenon, and one-hot data has a better answer (codes + ordered TS).
4. **The engine fix is a real core feature** (sparse column storage + skip-zero histogram fill + routing), not a preprocessing trick — it needs its own doc-17-style priced design. Until then the boundary is documented honestly: for genuinely sparse data above a few thousand columns, use lightgbm; bonsai's claims live on dense and categorical workloads.

Caveats: single corpus (rcv1), single split, fit-time on an M2 laptop (8 threads); xgboost's sparse AUC trails both — not investigated (out of scope).
