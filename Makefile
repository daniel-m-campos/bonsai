CAVEMAN_URL  := https://raw.githubusercontent.com/juliusbrussee/caveman/main/skills/caveman/SKILL.md
SKILLS_DIR   := .claude/skills
SOURCES      := $(shell find src include tests benchmarks -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) 2>/dev/null)
# src/python is excluded: the nanobind NB_MODULE macro and capsule-deleter
# FFI patterns trip cppcoreguidelines checks that don't apply to boundary code.
LINT_SOURCES := $(shell find src -type f -name '*.cpp' -not -path 'src/python/*' 2>/dev/null)
TOY_SENTINEL := tests/data/.toy-fetched

# Toolchain root: homebrew LLVM on macOS, apt.llvm.org layout on Linux.
# clang-tidy needs the macOS SDK sysroot spelled out; on Linux the driver
# finds its own sysroot.
ifeq ($(shell uname -s),Darwin)
LLVM_BIN       ?= /opt/homebrew/opt/llvm/bin
SDK_PATH       := $(shell xcrun --show-sdk-path)
LINT_EXTRA_ARGS := -extra-arg=-isysroot -extra-arg=$(SDK_PATH)
else
LLVM_BIN       ?= /usr/lib/llvm-21/bin
LINT_EXTRA_ARGS :=
endif

all: build

PYTHON ?= .venv/bin/python

build/build.ninja:
	@cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=$(LLVM_BIN)/clang++ \
	-G Ninja -S . -B build

configure: build/build.ninja

build: build/build.ninja
	@cmake --build build -j

build-cuda/build.ninja:
	@cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=$(LLVM_BIN)/clang++ \
	-DBONSAI_CUDA=ON \
	-G Ninja -S . -B build-cuda

build-cuda: build-cuda/build.ninja
	@cmake --build build-cuda -j

test-cuda: build-cuda $(TOY_SENTINEL)
	@ctest --test-dir build-cuda

build-asan/build.ninja:
	@cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_CXX_COMPILER=$(LLVM_BIN)/clang++ \
	-DBONSAI_SANITIZE=ON \
	-G Ninja -S . -B build-asan

build-asan: build-asan/build.ninja
	@cmake --build build-asan -j

test-asan: build-asan $(TOY_SENTINEL)
	@ctest --test-dir build-asan

clean:
	@rm -rf build build-cuda build-asan

rebuild: clean build

format:
	@$(LLVM_BIN)/clang-format -i $(SOURCES)

format-check:
	@$(LLVM_BIN)/clang-format --dry-run --Werror $(SOURCES)

# run-clang-tidy exits non-zero when findings exist; a non-zero exit with
# no findings means the tool itself failed and must not pass silently.
lint: build/build.ninja
	@log=$$($(LLVM_BIN)/run-clang-tidy -quiet -use-color=0 \
	    -clang-tidy-binary $(LLVM_BIN)/clang-tidy \
	    $(LINT_EXTRA_ARGS) \
	    -header-filter='include/bonsai/.*' -p build $(LINT_SOURCES) 2>&1); \
	status=$$?; \
	out=$$(echo "$$log" | grep -E '(warning|error):' \
	    | grep -v -E '(c\+\+/v1|system-headers|too many)'); \
	if [ -n "$$out" ]; then echo "$$out"; exit 1; fi; \
	if [ $$status -ne 0 ]; then echo "$$log" | tail -20; \
	    echo "lint: run-clang-tidy failed"; exit 1; fi; \
	echo "lint: no findings."

run: build
	@./build/src/bonsai $(ARGS)

test: build $(TOY_SENTINEL)
	@ctest --test-dir build

perf-benchmark: build
	@./build/benchmarks/bonsai_bench $(ARGS)

# Python extension. Needs a python with nanobind + numpy installed
# (override with PYTHON=/path/to/python).
python: build/build.ninja
	@cmake -B build -DBONSAI_PYTHON=ON -DBONSAI_OPENMP_STATIC=ON \
	    -DPython_EXECUTABLE=$(abspath $(PYTHON)) >/dev/null
	@cmake --build build --target _bonsai -j
	@echo "module at build/python/bonsai — use PYTHONPATH=build/python"

python-test: python $(TOY_SENTINEL)
	@PYTHONPATH=build/python $(PYTHON) python/tests/test_bindings.py

# CUDA-enabled extension in the CUDA tree; cuda_* growers can train.
python-cuda: build-cuda/build.ninja
	@cmake -B build-cuda -DBONSAI_PYTHON=ON -DBONSAI_OPENMP_STATIC=ON \
	    -DPython_EXECUTABLE=$(abspath $(PYTHON)) >/dev/null
	@cmake --build build-cuda --target _bonsai -j
	@echo "module at build-cuda/python/bonsai — use PYTHONPATH=build-cuda/python"

fit-benchmark: build $(TOY_SENTINEL)
	@uv run scripts/compare.py --config configs/california_housing.toml $(ARGS)

# GPU perf loop (benchmarks/README.md): MSD ladder vs xgboost-GPU,
# appends benchmarks/results/gpu_msd.jsonl. Needs the MSD dataset
# (scripts/fetch_year_msd.py) and a CUDA-capable host.
bench-gpu: build-cuda
	@BONSAI_CUDA_PROFILE=1 BONSAI_GROW_PROFILE=1 uv run scripts/bench_gpu.py $(ARGS)

$(TOY_SENTINEL):
	@uv run scripts/fetch_toy.py
	@touch $@

help:
	@echo "Targets:"
	@echo "  make build              Configure + compile."
	@echo "  make rebuild            Clean + build."
	@echo "  make test               Build + run ctest."
	@echo "  make build-cuda         Configure + compile with the CUDA backend (build-cuda/)."
	@echo "  make test-cuda          Build CUDA variant + run ctest against it."
	@echo "  make test-asan          Build ASan+UBSan variant (build-asan/) + run ctest."
	@echo "  make run ARGS=...       Build + run ./build/src/bonsai with ARGS."
	@echo "  make perf-benchmark     Build + run Catch2 perf microbenchmarks (ARGS forwarded)."
	@echo "  make fit-benchmark      Build + compare bonsai vs lightgbm/catboost on cal housing."
	@echo "  make bench-gpu          MSD ladder vs xgboost-GPU with profile breakdowns (GPU perf loop)."
	@echo "  make python             Build the Python extension (PYTHON=... to pick the interpreter)."
	@echo "  make python-cuda        Build the CUDA-enabled extension into build-cuda/python."
	@echo "  make python-test        Build + run python/tests/test_bindings.py."
	@echo "  make format             clang-format src/ + include/ + tests/ + benchmarks/ in place."
	@echo "  make format-check       clang-format --dry-run --Werror (CI gate)."
	@echo "  make lint               clang-tidy on src/ (header-filtered to bonsai)."
	@echo "  make skills             Install project-local Claude Code skills (currently: caveman)."
	@echo "  make skills-clean       Remove installed project-local skills."

skills: $(SKILLS_DIR)/caveman/SKILL.md

$(SKILLS_DIR)/caveman/SKILL.md:
	@mkdir -p $(@D)
	@echo "Fetching caveman skill -> $@"
	@curl -fsSL $(CAVEMAN_URL) -o $@
	@echo "Done. Restart Claude Code (or trigger a skill rediscovery) to pick it up."

skills-clean:
	rm -rf $(SKILLS_DIR)

.PHONY: configure build build-cuda build-asan clean rebuild format format-check lint all run test test-cuda test-asan perf-benchmark fit-benchmark bench-gpu python python-cuda python-test skills skills-clean help
