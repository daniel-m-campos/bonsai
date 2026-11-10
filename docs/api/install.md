# Install

`pip install bonsai-gbt` gives you the `bonsai` Python package and its compiled extension, packaged as a wheel (pip's prebuilt format). The `bonsai` command-line binary is not in the wheel: build from source to get it ([Building from source](building.md)).

## The one command

```bash
pip install bonsai-gbt
```

One command installs bonsai from [PyPI](https://pypi.org/project/bonsai-gbt/), with CPU training on every platform and GPU training on Linux x86_64. It supports Linux (x86_64, aarch64) and macOS (arm64) on CPython 3.9 through 3.13. The package installs as `bonsai-gbt` and imports as `bonsai`. GPU training needs only an NVIDIA driver. There is no separate GPU package and no CUDA toolkit to install, and the Linux x86_64 wheel is about 5.4MB. XGBoost's GPU wheel was about 300MB when measured for [decision 70](../decisions.md).

To install a specific release, run `pip install bonsai-gbt==1.7.0`. To install from the files attached to a GitHub release, run `pip install bonsai-gbt --find-links https://github.com/daniel-m-campos/bonsai/releases/expanded_assets/v1.7.0`. The [releases page](https://github.com/daniel-m-campos/bonsai/releases) lists every tag, and the wheels attach there too. One file per (CPython version, platform) combination attaches to each release, named `bonsai_gbt-<version>-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl` on Linux x86_64 (the aarch64 and macOS arm64 files follow the same pattern with their own platform tags, for example `cp312-cp312-macosx_14_0_arm64`), so you can install one by direct URL on a machine that cannot reach PyPI but can reach GitHub.

## What ships

| platform | architecture | GPU training | wheel |
|---|---|---|---|
| Linux | x86_64 | **yes** | `manylinux_2_34` (also tagged `2_35`), ~5.4MB |
| Linux | aarch64 | no (CPU) | `manylinux_2_34` (also tagged `2_35`) |
| macOS | arm64 | no (CPU) | `macosx_15_0` |

Every platform covers CPython 3.9 through 3.13. The macOS wheel needs macOS 15 or newer. The Linux wheels need glibc 2.34 or newer (Ubuntu 22.04 or Debian 12 and later). They bundle libc++ and the OpenMP runtime inside the wheel, so the only packages you install are numpy and typing_extensions, which pip adds for you.

## GPU support in the Linux x86_64 wheel

The x86_64 wheel includes the full CUDA backend ([decision 70](../decisions.md)). It has precompiled GPU code for NVIDIA architectures sm_70 (Volta), sm_75, sm_80, sm_86, sm_89, sm_90 and sm_120 (Blackwell consumer). It also embeds compute_90 PTX (NVIDIA's portable intermediate GPU code), which the driver compiles when the program loads, so GPUs newer than those, such as sm_100, still run. The CUDA runtime is statically linked, so you need no CUDA toolkit, only an NVIDIA driver at R525 or newer. On a machine without a GPU the same wheel behaves exactly like a CPU wheel: it imports, trains on CPU, and `bonsai.cuda_available()` reports `False`.

Every release's CUDA wheel is checked on a rented NVIDIA L40S GPU before it attaches to the release: the cuda_depthwise and cuda_levelwise growers (tree-growing strategies) each train 30 trees, their predictions must match the CPU depthwise and levelwise fits within 1e-3, and a saved model must reload with identical predictions (`scripts/wheel_smoke_cuda.py`, run by the `validate-cuda` job in `.github/workflows/wheels.yml`).

## Check it works

```python {.run}
import bonsai
import numpy as np

rng = np.random.default_rng(0)
X_train = rng.random((500, 8), dtype=np.float32)
y_train = (X_train[:, 0] * 2 + rng.normal(0, 0.1, 500)).astype(np.float32)

model = bonsai.BonsaiRegressor(n_iters=20).fit(X_train, y_train)
print("r2:", round(model.score(X_train, y_train), 3))
print("GPU available:", bonsai.cuda_available())
```

If `cuda_available()` is `True`, pass `device="cuda"` (the same argument XGBoost uses) to train on the GPU with the CUDA version of the default grower, the tree-growing strategy. To choose the strategy yourself, set `grower` to `"cuda_leafwise"`, `"cuda_depthwise"`, or `"cuda_levelwise"`. The [quickstart](quickstart.md) covers the rest.

## The bench extra

`pip install "bonsai-gbt[bench]"` also installs XGBoost (3.2 and 3.3 releases), LightGBM, CatBoost, pandas, openml, and nvidia-ml-py, the packages the benchmark scripts use. To rerun the benchmarks, follow [Reproducing the results](../results/benchmarks.md).

## Docker

A runtime image with the CUDA wheel preinstalled ships alongside each release:

```bash
docker run --gpus all ghcr.io/daniel-m-campos/bonsai:cuda python3 -c "import bonsai; print(bonsai.cuda_available())"
```

`bonsai:cuda` tracks the latest release; versioned tags (`bonsai:v1.7.0-cuda`) pin one. The image carries an sshd entrypoint keyed by a `PUBLIC_KEY` environment variable, so it boots directly as a RunPod (or similar) GPU pod. Before the image is promoted, the release gate boots this exact image on a rented L40S pod and runs the same checks.

## No wheel for your platform?

Wheels cover only the three platforms in the table above, so Windows, Intel macOS and macOS 14 or older have no wheel. The release page also carries the sdist (the source archive, `bonsai_gbt-<version>.tar.gz`), and `pip install` on it compiles bonsai on your machine. That build needs the LLVM 20+ and libc++ toolchain, so install it first. [Building from source](building.md) lists the supported Linux and macOS setups, and also covers the CLI binary and CUDA source builds.
