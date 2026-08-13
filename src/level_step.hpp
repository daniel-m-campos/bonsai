#pragma once

// The grower's data plane (docs/architecture/12-grower-backend.md, decision
// 41; transaction vocabulary from docs/architecture/14-engine-narrative.md,
// decision 53). LevelStep groups the per-tree data-plane steps — root setup,
// open_level (split finding), apply_level (row partitioning), child
// histogram construction, end_tree (leaf finalize) — behind one interface,
// selected by engine type: the primary template is the
// host plane (the CPU engine, and any engine's CPU fallback); the
// GPULevelEngine specialization is the device plane and holds the one runtime
// fork (on_device vs fallback) the design allows. The grow loops stay the
// control plane: every decision (leaf-vs-split, smaller-child pairing,
// constraint propagation) happens in grower_impl.hpp.
//
// The plane's pieces live in src/step/ and are included here, into the same
// translation unit: primitives.hpp (leaf writes, partition, the fill seam,
// the plan types), level.hpp (LevelStep), leaf.hpp (LeafStep).

#include "step/leaf.hpp"
#include "step/level.hpp"
#include "step/primitives.hpp"
