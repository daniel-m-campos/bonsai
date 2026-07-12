# Categorical trade-off probe, 2026-07: the evidence behind decision 58

The question (raised reviewing [architecture doc 17](../docs/architecture/17-categorical-splits.md)): is native categorical machinery in the core worth its complexity, or does preprocessing capture the gain? Method: toggle each reference library's own categorical machinery on/off at campaign-matched knobs (200 iters, lr 0.05, depth 6, 255 bins, seed 42, 80/20 split), and feed stock bonsai an ordered-target-statistics encoding built in ~100 lines of Python. Reproduce with `scripts/probe_categorical.py`; raw numbers in `results/cat-tradeoff-2026-07.json`.

Datasets: amazon employee access (OpenML 4135 — 32.8k×9, every column a high-cardinality ID; the native-categorical flagship), adult (1590 — 48.8k×14 mixed), kick / Don't Get Kicked (41162 — 73k×32, 18 categorical columns of mixed cardinality).

## Results (test AUC)

| variant | amazon | adult | kick |
|---|--:|--:|--:|
| bonsai, ordinal codes | 0.8307 | 0.9300 | **0.7830** |
| bonsai + OrderedTargetEncoder (TS + codes) | **0.8590** | **0.9306** | 0.7797 |
| bonsai + TS only, a=10 | 0.8536 | 0.9303 | 0.7792 |
| lightgbm, ordinal | 0.8278 | 0.9302 | 0.7846 |
| lightgbm, native set splits | 0.8572 | 0.9303 | 0.7666 |
| catboost, ordinal | 0.7791 | 0.9269 | 0.7740 |
| catboost, native (ordered TS + CTRs) | 0.8894 | 0.9283 | 0.7777 |
| xgboost, ordinal | 0.8052 | 0.9276 | 0.7827 |
| xgboost, native (`enable_categorical`) | 0.7878 | 0.9283 | 0.7791 |

## Readings

1. **Native Fisher set splits are a coin flip.** lightgbm's own toggle prices the feature at +0.029 (amazon), +0.0001 (adult), **−0.018** (kick — it overfits noisy high-cardinality columns). xgboost's native mode loses on two of three. This is the machinery doc 17 stage 2a would have added to the split scan, node format, routing, SHAP, and model format.
2. **The encoder beats the set splits where they win.** bonsai + ordered TS reaches 0.8590 on amazon — above lightgbm-native — from a preprocessing step users can decline per dataset (and should, on kick-shaped data).
3. **catboost's headline gain is partly self-rescue.** Its ordinal baseline (0.7791) is the worst number on amazon; measured against *bonsai's* baseline its true native edge is +0.059, of which the encoder closes 48%. The remainder is per-tree permutations plus crossed-category CTRs — not set splits — and crossed-TS preprocessing is the cheap next probe.
4. **On the repo's own amazon split** (`tests/data/amazon_*.csv`, the stage-1 files) the encoder measures +0.049 AUC (0.811 → 0.860), pinned in `python/tests/test_encoding.py`.

Caveats: single split per dataset, seed 42, no per-dataset tuning — same convention as the quality campaign (`quality-campaign-2026-07.md`); AUC differences under ~0.003 are chance-band at these test sizes.
