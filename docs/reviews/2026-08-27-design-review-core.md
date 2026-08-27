# Design review: the core

2026-08-27, on Daniel's request. Lens: SOLID plus data-oriented design plus DRY, composition over inheritance, and least knowledge, applied to the C++ core (`include/bonsai/`, `src/`). Scope excludes the Python layer (reviewed 2026-07-30) and the CUDA kernels (reviewed 2026-07-03). The three findings below shipped as fixes in the same PR, one commit each.

## Scoping

Extensibility pressure: high. Not from headcount but from variant count: three growers, seven objectives, three samplers, two engines, two planes, and a registry built so a fourth of any of them is additive.

Performance pressure: high. Fits at 16M x 128 where the histogram fill dominates; the repo prices changes through a perf ritual and a standings ledger.

Data clarity: clear. float32 matrix plus labels, to binned columns (`BinStore`), to per-node (grad, hess) cells, to splits, to trees, to scores. Each module states its input and output.

That quadrant calls for compile-time seams at the boundaries and data-oriented interiors, and that is what the code does: the seams are C++20 concepts (`TreeGrower`, `HistogramEngine`, `Objective`, `Sampler`), zero-cost at instantiation, while the interiors are spans, arena-carved histograms, and plain structs. Virtual dispatch appears at exactly one place, the type-erasure boundary where the CLI and Python hold a model they did not instantiate. The fit path contains no virtual calls.

## Findings, each fixed in this PR

**1. `IBooster` coupled two clients to methods each never calls.** Measured, not asserted: of its 24 virtuals, `seed_validation_scores`, `accumulate_last_round`, `accumulate_last_round_binned`, `validation_loss`, `begin_resident_validation`, `accumulate_last_round_resident`, and `truncate` were referenced only by `src/cli/pipeline.cpp`, while `device_plan_input`, `predict_staged`, and `predict_leaf` only by `src/python/module.cpp`. The interface's own section comment said so ("the training-loop seam, CLI pipeline only"); the comment was the finding. Fixed: `ITrainableBooster : IBooster` carries `update_one_iter` plus the seven-method seam. A derived interface rather than two siblings because the pipeline genuinely uses both halves, while the Python `Model` holds the base and receives it by upcast from the same factory products. `score_width` stays on the base, both clients size buffers from it.

**2. The growers wrote the device seam three times.** Each grower carried `engine_`, `resident_`, `eval_` and five methods managing them, about forty lines apiece, near verbatim; the only real difference was leafwise arming through the pool-sizing call. Fixed: `DeviceSeam<EngineT>` owns the engine and both arming flags, with the seam's contract comment stated once on it (the `RecycledOutputs` pattern). The growers keep their public methods as one-line forwarders, so the booster and every test see an unchanged surface. The three `eval_accumulate` definitions stay per-grower: each tree shape flattens to a device node table its own way, and after the dedup that lambda is the only content left in them.

**3. `Booster::update_one_iter` encoded a load-bearing order as statement order.** Renewal overwrites every leaf value, the monotone reprojection restores the constraint over what renewal wrote, and the score update reads the final values. Nothing named that sequence, which is how a violation of it went unnoticed: leaf renewal silently discarded the monotone bounds on the node-splitting growers for as long as both features have coexisted (issue #442). Fixed: the legs after the grower returns move into `finish_round`, whose contract comment states the order and why it is the contract.

## What held up under attack

`Dataset` is the strongest design in the tree. A view costs a row descriptor rather than a matrix; eight accessors forward through the facade so call sites read `ds.cuts(f)` and never chain through the store; and the `LabelsId`/`FitId` strong enums close a real bug class, stated on the type: a freed block's address comes back from the allocator, so a cache keyed on it would serve the previous fit's data to a same-shaped successor. Equality on a minted token cannot.

`sampler_traits` is the open/closed exemplar. The primary template answers conservatively (`copies_view_verbatim = false`, `reads_gradients = true`), so a sampler added later gets the correct slow path by default and opts into the fast paths deliberately, instead of being forgotten in a chain of type comparisons where the omission is a silent wrong answer.

`HistCell` is array-of-structs and correctly so: every fill and every split scan touches both fields together, so structure-of-arrays would double the memory streams for no locality gain. The float-cell, double-reduction split prices storage bandwidth and accuracy separately.

Inheritance depth is two everywhere and exists only at the type-erasure boundary; variation inside is templates, concepts, and traits. No method chains beyond two hops anywhere in the core.

Two apparent violations were checked and correctly left alone. `src/python/module.cpp` at 2,206 lines looks like a DRY violation, but its split was priced and declined on measurement (the nanobind floor is 0.43s, and 69% of its churn sits in four coupled things). `SoftmaxObjective` genuinely violates substitutability, a dispatch tag wearing the `Objective` interface whose 1-D `eval` cannot express K columns and throws, but the alternative, a parallel tag hierarchy, would need the same stubs under a different name to satisfy the same trait tables; the trade is recorded on the type itself.

## Grades

| principle | grade | note |
| --- | --- | --- |
| single responsibility | B | clean cuts at grower/engine and Dataset/BinStore; `Booster` still owns seven concerns, two now named (`finish_round`) |
| interface segregation | C, fixed to A- | the one measured violation, resolved by the seam split |
| dependency inversion | A | Config and template injection throughout |
| open/closed | A | registry, name traits, conservative trait defaults |
| substitutability | C | `SoftmaxObjective` and the throwing `predict_proba` defaults, both documented trades |
| data-oriented design | A | layout chosen from access patterns, costs priced, no speculative abstraction found |
| DRY | B, fixed to A- | the grower seam was the one real instance |
| composition over inheritance | A | depth two, virtual only where runtime selection is real |
| least knowledge | A | the Dataset facade exists specifically to prevent chains |

## Residue

`Booster` remains the largest class by responsibility count: the boosting loop, the score cache, DART, renewal and reprojection (now grouped), resident arming, and the densify cache. The next candidates for extraction are `apply_dart_round` with its drop bookkeeping, and the resident round pair. Neither is urgent; both would follow the `finish_round` pattern.

`predict_proba`'s throwing default on the base interface narrows the contract for every width-1 booster. The honest alternative is a `IProbabilisticBooster` extension the multiclass booster alone implements, at the cost of a `dynamic_cast` in the Python module. Deferred: the throw is documented, tested, and reachable only through a caller that ignored `score_width`.
