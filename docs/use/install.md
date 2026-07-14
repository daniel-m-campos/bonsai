# Install

The wheel ships the `bonsai` Python package and its compiled extension, nothing else: the `bonsai` command-line binary is a source-build artifact ([Building from source](building.md)). There is no PyPI listing yet; wheels attach to every [GitHub release](https://github.com/daniel-m-campos/bonsai/releases).

## The one command

```bash
pip install bonsai-gbt --find-links https://github.com/daniel-m-campos/bonsai/releases/expanded_assets/v1.3.0
```

`--find-links` points pip at a plain HTML listing of the release's assets; pip scans it and picks the wheel matching your Python and platform, exactly as it would on a package index. The URL pins a release tag, so bump it when a new version ships (the [releases page](https://github.com/daniel-m-campos/bonsai/releases) always has the latest).

## What ships

| platform | architecture | GPU training | wheel |
|---|---|---|---|
| Linux | x86_64 | **yes, out of the box** | `manylinux_2_34`, ~2.3MB |
| Linux | aarch64 | no (CPU) | `manylinux_2_34` |
| macOS | arm64 | no (CPU) | `macosx_14_0` |

Every platform covers CPython 3.9 through 3.13. The Linux tags mean Ubuntu 22.04+/Debian 12+ era glibc; libc++ and the OpenMP runtime are vendored into the wheel, so nothing needs installing beside numpy (which pip pulls automatically).

## GPU support in the linux x86_64 wheel

The x86_64 wheel carries the CUDA backend whole ([decision 70](../decisions.md)): native code for every real NVIDIA architecture from sm_70 (Volta) through sm_120 (Blackwell consumer), plus a compute_90 PTX floor that forward-JITs on anything newer. The CUDA runtime is statically linked, so you need no CUDA toolkit, only an NVIDIA driver at R525 or newer. On a machine without a GPU the same wheel behaves exactly like a CPU wheel: it imports, trains on CPU, and `bonsai.cuda_available()` reports `False`.

Every release's CUDA wheel is validated on rented GPU hardware before it attaches to the release; the claim "pip install, then GPU training works" is tested, not hoped.

## Check it works

```python
import bonsai

model = bonsai.BonsaiRegressor(n_iters=20).fit(X_train, y_train)
print("r2:", round(model.score(X_train, y_train), 3))
print("GPU available:", bonsai.cuda_available())
```

If `cuda_available()` is `True`, pass `grower="cuda_depthwise"` (or `"cuda_oblivious"`) to train on the GPU; the [API tour](api-tour.md) covers the rest.

## The bench extra

```bash
pip install "bonsai-gbt[bench]" --find-links https://github.com/daniel-m-campos/bonsai/releases/expanded_assets/v1.3.0
```

The extra pulls xgboost, lightgbm, catboost, scikit-learn, pandas, and openml: everything `bonsai.bench` needs to reproduce the published benchmark tables. `python -m bonsai.bench.grinsztajn out.jsonl --report` re-runs the external standings suite under the [benchmark protocol](../method/benchmark-protocol.md).

## Docker

A runtime image with the CUDA wheel preinstalled ships alongside each release:

```bash
docker run --gpus all ghcr.io/daniel-m-campos/bonsai:cuda python3 -c "import bonsai; print(bonsai.cuda_available())"
```

`bonsai:cuda` tracks the latest release; versioned tags (`bonsai:v1.3.0-cuda`) pin one. The image carries an sshd entrypoint keyed by a `PUBLIC_KEY` environment variable, so it boots directly as a RunPod (or similar) GPU pod. The release gate validates this exact image on real hardware before promoting it.

## Picking a wheel by hand

For locked-down environments or requirements files, install a release asset by direct URL:

```bash
pip install https://github.com/daniel-m-campos/bonsai/releases/download/v1.3.0/bonsai_gbt-1.3.0-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl
```

Filename anatomy: `cp312` is the CPython version (pick yours, 39 through 313), the platform tag is `manylinux_*_x86_64` / `manylinux_*_aarch64` / `macosx_14_0_arm64`, and there is one file per combination on the release page.

## No wheel for your platform?

The sdist (`bonsai_gbt-<version>.tar.gz`) is on the release page and `pip install` will compile it, but that is a full source build: it needs the LLVM/libc++ toolchain described in [Building from source](building.md), which is also where development setups, the CLI binary, and CUDA source builds live.
