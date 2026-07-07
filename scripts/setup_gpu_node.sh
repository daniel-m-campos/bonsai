#!/usr/bin/env bash
# One-shot bonsai CUDA setup for a fresh x86_64 Ubuntu 22.04 GPU node
# (validated on Thunder Compute 4x A100 standard VMs — use standard VMs
# only; 1x/2x instances virtualize the GPU and are unusable for this).
# Usage: bash setup_gpu_node.sh [branch]   (default: cuda-phase2)
# Idempotent: safe to re-run. See benchmarks/README.md for the workflow.
set -euxo pipefail

BRANCH="${1:-cuda-phase2}"
export PATH="$HOME/.local/bin:$PATH"

# Thunder's telemetry shim (/etc/ld.so.preload) busy-spins when many
# CUDA-linked processes start concurrently (ctest -j, parallel builds).
# The GPUs are locally attached on standard VMs; nothing needs the shim.
if [ -s /etc/ld.so.preload ]; then
    sudo mv /etc/ld.so.preload /etc/ld.so.preload.disabled
fi

# --- toolchain: LLVM 21 + libc++ (apt.llvm.org; add-apt-repository is
# broken on the lean standard-VM image, so write the source line directly)
if [ ! -x /usr/lib/llvm-21/bin/clang++ ]; then
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null
    echo "deb https://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" | sudo tee /etc/apt/sources.list.d/llvm21.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq clang-21 libc++-21-dev libc++abi-21-dev libomp-21-dev
fi
sudo apt-get install -y -qq ninja-build git

# --- uv (also provides cmake: the image ships 3.22, below our 3.25 floor)
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -q
command -v cmake >/dev/null 2>&1 && [ "$(cmake --version | head -1 | grep -oE '[0-9]+' | head -1)" -ge 4 ] || uv tool install -q cmake

# --- CUDA 12.6 side-by-side (clang-21 cannot target CUDA 13)
if [ ! -x /usr/local/cuda-12.6/bin/nvcc ]; then
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-6
fi

# --- repo + build
[ -d ~/bonsai ] || git clone -q https://github.com/daniel-m-campos/bonsai.git ~/bonsai
cd ~/bonsai
git fetch -q origin
git checkout -q "$BRANCH"
git pull -q --ff-only || true
cmake -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/lib/llvm-21/bin/clang++ \
    -DBONSAI_CUDA=ON \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.6
ninja -C build-cuda

# --- datasets
uv run --quiet scripts/fetch_year_msd.py
uv run --quiet scripts/fetch_toy.py

set +x
echo "SETUP-COMPLETE"
echo "next: cd ~/bonsai && make bench-gpu   (or: ctest --test-dir build-cuda -j16)"
