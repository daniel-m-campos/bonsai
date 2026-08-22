# bonsai-ci: the pinned toolchain for RunPod validation pods and any Linux
# perf work. Kills the per-pod setup roulette (apt.llvm.org fetches, CUDA
# image/driver mismatches, museum cmake) by baking everything a validation
# session needs. Built and pushed by .github/workflows/ci-image.yml to
# ghcr.io/daniel-m-campos/bonsai-ci.
#
# CUDA 12.8 base: the floor for sm_120 (Blackwell) --offload-arch=native.
# r550 fleet (L40S) runs 12.8-built binaries via minor-version compatibility.
FROM nvidia/cuda:12.8.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        ca-certificates curl wget gnupg git openssh-server ninja-build \
        lsb-release software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm.sh | bash -s -- 21 \
    && apt-get install -y -qq clang-21 libc++-21-dev libc++abi-21-dev libomp-21-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv provides python 3.12 (nanobind ABI pin) and a modern cmake; jammy's apt
# cmake (3.22) predates the project's minimum.
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ~/.local/bin/uv python install 3.12
ENV PATH="/root/.local/bin:${PATH}"
# xgboost bounded below 3.4: its 3.4.0 wheel ships a CUDA 13 runtime that
# silently falls back to CPU on this image's r550-and-older driver hosts.
# nvidia-ml-py gives the bench VRAM sampler per-pid attribution via NVML, which
# works inside containers where nvidia-smi's PID namespace does not.
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        cmake ninja numpy "nanobind<3" scikit-learn pandas tabulate matplotlib \
        "xgboost>=3.2,<3.4" catboost nvidia-ml-py
ENV PATH="/opt/venv/bin:${PATH}"

# lightgbm from source with the CUDA backend: the PyPI wheel is CPU-only,
# and the lgbm_cuda reference arm needs device_type=cuda on pods. The same
# install serves lgbm_cpu (device_type=cpu works in a CUDA build). Compiling
# needs no GPU; sm_89/sm_80 pods run it via the build's own arch list.
RUN CUDACXX=/usr/local/cuda/bin/nvcc \
    uv pip install --python /opt/venv/bin/python --no-binary lightgbm \
        --config-setting cmake.define.USE_CUDA=ON lightgbm

# FetchContent sources, pre-cloned: pods point CMake at these via
# FETCHCONTENT_SOURCE_DIR_* instead of hitting GitHub per configure.
RUN mkdir -p /opt/deps \
    && git clone -q --depth 1 --branch v3.5.4 https://github.com/catchorg/Catch2.git /opt/deps/Catch2 \
    && git clone -q --depth 1 --branch v2.6.2 https://github.com/CLIUtils/CLI11.git /opt/deps/CLI11 \
    && git clone -q --depth 1 --branch v3.4.0 https://github.com/marzer/tomlplusplus.git /opt/deps/tomlplusplus \
    && git clone -q --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git /opt/deps/json

# RunPod-compatible entrypoint: installs the PUBLIC_KEY env into
# authorized_keys and runs sshd in the foreground for direct-IP SSH access.
# Known limitation: ssh.runpod.io proxy routing needs RunPod's in-image
# agent (official images only), so pods on this image are reachable only
# via publicIp:portMappings['22'] — machines that expose no public IP need
# re-rolling. bench workers additionally need PYTHONPATH=build/python.
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
