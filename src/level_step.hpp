#pragma once

// The grower's data plane (decisions 41 and 53). LevelStep groups the per-tree
// data-plane steps — root setup, open_level (split finding), apply_level (row
// partitioning), child histogram construction, end_tree (leaf finalize) — behind one
// interface, selected by engine type: the primary template is the host plane and the
// GPULevelEngine specialization is the device plane. Neither carries a runtime
// fork: a tree the device cannot hold is refused, not moved. The grow loops
// stay the control plane: every decision (leaf-vs-split, smaller-child
// pairing, constraint propagation) happens in grower_impl.hpp.
//
// The plane's pieces live in src/step/ and are included here, into the same
// translation unit: primitives.hpp (leaf writes, partition, the fill seam,
// the plan types), level.hpp (LevelStep), leaf.hpp (LeafStep).

#include "step/leaf.hpp"
#include "step/level.hpp"
#include "step/primitives.hpp"
