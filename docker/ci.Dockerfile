# bonsai-ci: the pinned toolchain for RunPod validation pods and any Linux
# perf work. Kills the per-pod setup roulette (apt.llvm.org fetches, CUDA
# image/driver mismatches, museum cmake) by baking everything a validation
# session needs. Built and pushed by .github/workflows/ci-image.yml to
# ghcr.io/daniel-m-campos/bonsai-ci.
#
# CUDA 12.8 for Blackwell (sm_120): clang refuses --offload-arch=native
# under a 12.4 toolkit for sm_120 ("GPU arch sm_120 is supported by CUDA
# versions between 12.8 and..."), which the redesigned RTX PRO 6000
# standings host hit on its maiden pod. The image now builds for the
# fleet's full driver range, r550 L40S through r580 Blackwell: CUDA
# minor-version compatibility lets 12.8-built binaries run against r550
# drivers, the same mechanism the xgboost pin below already relies on.
# Published as the cuda12.8 tag, leaving cuda12.4 frozen for branches
# still pinned to it. The xgboost pin rationale is unchanged by this
# bump: 3.3 stays a CUDA 12.9 wheel, so the CUDA-13 bound below still
# binds; only a deliberate move to a CUDA 13 base would flip it.
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
        cmake ninja numpy nanobind scikit-learn pandas tabulate matplotlib \
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
