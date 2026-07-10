# bonsai-ci: the pinned toolchain for RunPod validation pods and any Linux
# perf work. Kills the per-pod setup roulette (apt.llvm.org fetches, CUDA
# image/driver mismatches, museum cmake) by baking everything a validation
# session needs. Built and pushed by .github/workflows/ci-image.yml to
# ghcr.io/daniel-m-campos/bonsai-ci.
#
# CUDA 12.4 deliberately: every RunPod machine we've rented satisfies a 12.4
# image, while cu12.8+ images silently fail to start on 12.4-driver hosts.
# The A100/L40S targets (sm_80/sm_89) need nothing newer.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        cmake ninja numpy nanobind scikit-learn pandas lightgbm
ENV PATH="/opt/venv/bin:${PATH}"

# FetchContent sources, pre-cloned: pods point CMake at these via
# FETCHCONTENT_SOURCE_DIR_* instead of hitting GitHub per configure.
RUN mkdir -p /opt/deps \
    && git clone -q --depth 1 --branch v3.5.4 https://github.com/catchorg/Catch2.git /opt/deps/Catch2 \
    && git clone -q --depth 1 --branch v2.6.2 https://github.com/CLIUtils/CLI11.git /opt/deps/CLI11 \
    && git clone -q --depth 1 --branch v3.4.0 https://github.com/marzer/tomlplusplus.git /opt/deps/tomlplusplus \
    && git clone -q --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git /opt/deps/json

# RunPod-compatible entrypoint: installs the PUBLIC_KEY env into
# authorized_keys and runs sshd in the foreground, matching what the
# official runpod images' start scripts do for direct-IP SSH access.
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
