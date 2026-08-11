# The archive

This page is hand-maintained. Nothing computes it: the rows are frozen prose, and the data behind each one lives in git history.

Every campaign and probe below is closed. Its verdict is frozen in [the decisions log](../../decisions.md), its narrative is in [`benchmarks/`](../../../benchmarks), and its data file was deleted from `benchmarks/results/` rather than kept beside current evidence, which is the results-lifecycle policy (decision 92): the ledger renders what is current, whole, and git history keeps what is not.

The last column is the last commit where the file was present, so the rows still read:

```
git show <ref>:benchmarks/results/<file>
```

| record | what it found | decisions | data files | last present at |
|---|---|---|---|---|
| The retired row and width standings | The last measurement on the rows and width axes the scenario redesign replaced: 16M x 100 and the column sweep to 65k features, both planes, on one pod. | 103 | `rebaseline-2026-08.jsonl`, `cols-rebaseline-2026-08.jsonl` | `9221cf7` |
| The retired shape and frontier standings | The iso-volume aspect sweep at 2^31 cells and the 16M accuracy-versus-time frontier, both replaced by the tall, wide, and extreme scenarios. | 91, 103 | `iso-volume-2026-08.jsonl`, `gpu-pareto-16M-2026-08.jsonl` | `a32eccf` |
| The retired airline standings | The one real-data perf axis, removed with the redesign: its 10M-row CSV bake answered a protocol question that no longer has a scenario. | 103 | `airline-2026-08.jsonl` | `9221cf7` |
| The retired early-stop axis | Early stopping measured at 4M rows, superseded by the gpu-early-stop axis at the tall cell. | 103 | `early-stop-4M-2026-08.jsonl` | `89814a0` |
| The XGBoost 3.3 recheck | 3.3 matched 3.2 at every GPU cell and did not move host RSS; its one real gain was wide-CPU histogram time at 1M x 4096, and no published cell flipped. | 87 | `xgb33-recheck-2026-07.jsonl` | `563879a` |
| The CPU 16M round | Software prefetch closed the 16M CPU gap to XGBoost-hist on that pod. | 61 | `cpu-prefetch-round-2026-07.jsonl` | `4b40fd9` |
| Ordered boosting at scale (the CatBoost door) | CatBoost's Ordered and Plain modes against bonsai levelwise as rows grow: the door that closed the small-data-edge question. | 62 to 64 | `catboost-scale-edge-2026-07.jsonl` | `7b0e1ac` |
| The leafwise recheck | Decision 42's reading, bonsai's CPU leafwise ahead of LightGBM's CUDA leaf-wise, inverts monotonically with scale to 5.3x in LightGBM's favor at 16M. | 95 | `leafwise-recheck-2026-08.jsonl` | `54497c1` |
| The device-leafwise cadence probe | The round skeleton's fixed cost measured 30.3 us against a 100 us budget, which is what let the one-node-per-round design proceed to a build. | issue #268 | `leafwise-cadence-2026-08.json` | `267b705` |
| The device-leafwise admission ladder | The pre-registered kill criterion met: `cuda_leafwise` beat LightGBM's CUDA leaf-wise at 16M x 100 on the same pod, and beat its own CPU arm at every cell. | 97 | `leafwise-ladder-2026-08.jsonl` | `db74fb3` |
| The closing leafwise ladder | Stage 3's two levers, the device-resident objective and the round's pinned staging, cut leafwise fit at every cell; a third was refuted and reverted. | 98 | `leafwise-stage3-2026-08.jsonl` | `a5a2e9d` |
| The leafwise correction | Both campaign ladders re-measured on the fixed ingest path: the harness had binned on the host, so every published bonsai CUDA time was slow for a reason that was never the library. | 100 | `leafwise-correction-2026-08.jsonl` | `e4eb2b9` |
| The wide-CPU fill | The row-wise u8 fill missed cache at 16k features; routing feature-parallel fixed the symptom and one column-block-tiled pass retired the strategy pair. | 88, 89 | `wide-cpu-hist-2026-07.jsonl` | `228e2e4` |
| The CUDA wide recheck | The recorded wide-GPU gap to XGBoost no longer existed on current main, so the campaign closed at stage 0 and the stale reading was corrected wherever it appeared. | 90 | `cuda-wide-recheck-2026-07.jsonl` | `b60cecf` |
| The single-card ceiling | A 500M x 100 float32 matrix trained end to end on one 80GB card: 60 rounds in 8.5 minutes at 69.9 GiB peak device memory. | engine chapter 5 | `single-card-ceiling-2026-07.jsonl` | `7e9c8fd` |
| Campaign smoke: ten datasets at matched knobs | The fast local regression check behind the quality campaign's three model-changing fixes. | 56, 57 | `quality-campaign-2026-07.jsonl` | `1f8eb20` |
| Probe: per-feature bin budgets | No bin-budget policy moved a standing outside the chance band, so the 255-bin default stayed. | 67 | `binning-probe-2026-07.json` | `98fab49` |
| Probe: categorical machinery | The categorical question resolved as an encoder rather than per-split machinery in the core. | 58 | `cat-tradeoff-2026-07.json` | `7465ede` |
| Probe: ranking objectives | The stable NDCG@10 gap is to listwise losses only, which is how the ranking issue is scoped. | issue #58 | `ranking-tradeoff-2026-07.jsonl` | `95f7eca` |
| Probe: CatBoost's categorical toggle | CatBoost's own toggle prices its categorical machinery at 68% of its remaining cat-heavy lead over bonsai's encoder. | 80 | `tabarena-cat-probe-2026-07.jsonl` | `0efa5ce` |
| Probe: ordered boosting | Ordered beat Plain beyond the chance band on 0 of 12 datasets, at 3.9x the train time. | 81 | `ordered-boosting-probe-2026-07.jsonl` | `885db11` |
| Probe: static K-permutation target statistics | K-averaged ordered statistics recover a negative share of the gap: the substance is per-split, not preprocessing. | 82 | `static-k-encoder-probe-2026-07.jsonl` | `66f587f` |
| Probe: a per-dataset learning-rate rule | A validation-selected oracle over eight rates wins validation and loses test, and CatBoost's automatic rate transplants to a no-op around the shipped 0.05. | 83 | `lr-rule-probe-2026-07.jsonl` | `974f5c6` |
| Probe: the bagged-protocol randomization interaction | CatBoost's small-data lead is not a bagging interaction: 8-fold data-bagging already gives bonsai the decorrelation. | 85 | `bagging-interaction-probe-2026-07.jsonl` | `9c3ceaa` |
| Probe: honest shadow-feature selection | The shadow selector sits inside the chance band against plain top-k on 26 of 27 grower-dataset cells, under all three growers. | 86 | `feature-selection-probe-2026-07.jsonl` | `981dc1c` |
| The selection-method survey | Ten selection methods, one shared judge, refit down a budget ladder on an untouched holdout; the worked example in guide chapter 14. | 86 | `selection-survey-2026-07.jsonl` | `fa6838a` |
| The Grinsztajn min_child_weight bracket | XGBoost's ambiguous knob translation measured at both ends; the standings order holds under either convention. | 68 | `grinsztajn-2026-07-xgb-mcw1.jsonl` | `5e45b14` |
