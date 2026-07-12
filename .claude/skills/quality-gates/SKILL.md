---
name: quality-gates
description: >
  bonsai's gate ritual for any change to training code. Use before/after
  every commit that touches src/ or include/ — especially refactors that
  claim byte-identical behavior. Invoke via /quality-gates or whenever a
  change set needs validation.
---

Run every gate; a change that claims behavior preservation must pass all of them.

## Before touching anything

Capture the byte-identity baseline FIRST (a baseline taken after edits proves nothing):

```bash
make python && PYTHONPATH=build/python python3 scripts/model_hash.py
```

Record the printed `sha256:` value. As of 2026-07 the fabel baseline is `97a7418f3b0f11ed`, but always capture fresh — the hash legitimately changes when a decision-gated model change lands.

## Per commit

```bash
make format          # clang-format via the pinned llvm@21 toolchain
make build && ctest --test-dir build -j8
make lint            # run-clang-tidy; scope: src/*.cpp + include/bonsai
PYTHONPATH=build/python python3 scripts/model_hash.py   # must equal the baseline
```

Rules learned the hard way:
- **Never `git add -A` after `make format`** if homebrew LLVM has drifted past the pin — v22 reformats of untouched files once rode into a commit. Stage explicitly.
- The Mac builds the CUDA **stub**; `.cu` changes compile only in CI's `cuda-compile` job and run only on a pod. A green Mac suite says nothing about kernels — see /pod-validate.
- A hash match with recycled/reused buffers is a *strong* result (tree n+1 starts from tree n's bytes); a mismatch on a "pure refactor" means the refactor isn't pure — find out why before rationalizing.
- Model-changing work (new cuts, new objectives' defaults) gets a decision entry in docs/decisions.md with before/after quality numbers instead of the hash gate.
