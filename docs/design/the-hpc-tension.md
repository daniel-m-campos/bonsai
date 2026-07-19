# The HPC tension

The seams in [the system map](system-map.md) are concepts, one type per backend.
This page is where those seams meet performance: the GPU.
The honest story is that a concept can only check so much.
The rest is held by tests, and by rules the code refuses to break.

## The concept checks two signatures; determinism is the real contract

`HistogramEngine` requires exactly two methods, `begin_tree` and `populate` ([Concepts to types](api-tour-concepts.md) quotes it).
The load-bearing contract is not in the requires-clause.
It is in the comment above the concept in [`grower.hpp`](../../include/bonsai/grower.hpp): `populate` must accumulate the node's rows into the bins the mappers define, in "an order that is a pure function of configuration," with missing values in the last bin.
A type can satisfy the syntax and bend any of those.
The comment names the consequence: it "trains silently wrong models."
The compiler cannot see this, but the `[cuda]` parity suite can.
It asserts CPU and GPU agree within `1e-4` ([`tests/unit/test_cuda_grower.cpp`](../../tests/unit/test_cuda_grower.cpp)).
The contract lives in the suite, and the concept is only its syntactic floor.

## One concept, not seven, because the device works whole or not at all

The GPU LevelStep is a single concept, `GPULevelEngine`, not one concept per stage.
Its comment gives the reason: the LevelStep "drives this whole cluster or none of it," so it is one concept and not seven.
Device histograms, rows, and split finding are coupled to the same resident state ([grower-backend doc](../architecture/12-grower-backend.md)).
So they cannot be mixed with host stages.
`begin_root` returns `false` when the resident buffers will not fit, and the grow loop then runs that whole tree on the host plane.

## The planes divide state by lifetime

The device context splits its resident memory into four planes, each with its own lifetime ([`src/cuda/detail/device_context.cuh`](../../src/cuda/detail/device_context.cuh)):

| Plane | Holds | Lifetime |
|---|---|---|
| `DeviceData` | the binned matrix | once per fit |
| `GradientPlane` | per-tree gradients, interleaved to `float2` | once per tree |
| `LevelPipeline` | rows, histograms, staging buffers | once per level |
| `ResidentPlane` | labels, scores, and resident-objective state | once per fit |

Dividing by lifetime is what keeps an edge honest: an upload done once per fit must never be redone per tree.
Naming the planes also makes the boundary crossings countable.
That is how the [compute DAG](../architecture/16-compute-dag.md) prices a move before it is played.

## The resident objective deleted a boundary instead of optimizing across it

For MSE, LogLoss, or Poisson with all-rows or Bernoulli sampling, the whole per-tree host round trip disappears.
Labels and scores upload once into `ResidentPlane`.
Each tree derives its gradients on the card, and the epilogue folds the leaf values back into the resident scores.
The lesson from [case E4](../learn/engine/4-the-resident-objective.md) is the title of this section: delete a boundary, do not optimize across it.
The single-GPU 16M oblivious round fell from 104 to 64 ms (decision 78).
The resident model proved bit-identical to the host-objective model on a Jetson.

## What stays host-side on purpose

The control plane stays on the host by design ([grower-backend doc](../architecture/12-grower-backend.md)).
Split decisions cross the bus down every level, because the grow loop must observe each level's outputs before opening the next.
That pins one small device-to-host sync per level, the irreducible floor of about 800 syncs per fit.
On a healthy host each costs 10 to 20 microseconds ([compute DAG](../architecture/16-compute-dag.md)).
Mapper-fit stays host-side too: its cut points come from a seeded RNG stream.
Reproducing that stream on the device would risk the determinism identity for no gain.

## The honest cost

The seams forbid two things a looser design would allow.
There is no per-node GPU fallback.
An engine either grows a whole tree resident or declines it in `begin_root`.
A single oversized feature then sends the tree to the CPU plane ([GPU-resident doc](../architecture/11-gpu-resident.md)).
There are no cross-plane shortcuts: a placement move must not change accumulation order or precision, and the byte-identity gate catches it if it does.

What that bought:

- The parity suite: one contract, checked at `1e-4`, that any engine must meet.
- Bit-identical CPU models across architectures and thread counts, checked per commit ([determinism](determinism.md)).
- The `1e-4` GPU convention: `cuda_*` models are tolerance-equal, not tree-equal, because atomic add order differs. The tests assert prediction tolerance, never tree equality ([GPU-resident doc](../architecture/11-gpu-resident.md)).

The trade is stated plainly.
The device cannot cut through the seams.
In exchange, every model bonsai ships is either bit-reproducible, or provably within `1e-4` of the model that is.
