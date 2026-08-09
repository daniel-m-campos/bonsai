CAVEMAN_URL  := https://raw.githubusercontent.com/juliusbrussee/caveman/main/skills/caveman/SKILL.md
SKILLS_DIR   := .claude/skills
SOURCES      := $(shell find src include tests benchmarks -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) 2>/dev/null)
# src/python is excluded: the nanobind NB_MODULE macro and capsule-deleter
# FFI patterns trip cppcoreguidelines checks that don't apply to boundary code.
LINT_SOURCES := $(shell find src -type f -name '*.cpp' -not -path 'src/python/*' 2>/dev/null)
TOY_SENTINEL := tests/data/.toy-fetched
AMAZON_SENTINEL := tests/data/.amazon-fetched

# Toolchain root: homebrew LLVM on macOS, apt.llvm.org layout on Linux.
# clang-tidy needs the macOS SDK sysroot spelled out; on Linux the driver
# finds its own sysroot. CI pins LLVM 21; prefer the matching versioned keg
# when installed — homebrew's mainline llvm drifted to 22, whose stricter
# pedantic set (-Wc2y-extensions on Catch2's __COUNTER__) and reformats
# break the repo. The llvm@21 bottle's libc++ also mislabels float
# from_chars availability on macOS, hence the define.
ifeq ($(shell uname -s),Darwin)
ifneq ($(wildcard /opt/homebrew/opt/llvm@21/bin),)
LLVM_BIN       ?= /opt/homebrew/opt/llvm@21/bin
export CXXFLAGS := $(CXXFLAGS) -D_LIBCPP_DISABLE_AVAILABILITY
else
LLVM_BIN       ?= /opt/homebrew/opt/llvm/bin
endif
SDK_PATH       := $(shell xcrun --show-sdk-path)
LINT_EXTRA_ARGS := -extra-arg=-isysroot -extra-arg=$(SDK_PATH)
else
LLVM_BIN       ?= /usr/lib/llvm-21/bin
LINT_EXTRA_ARGS :=
endif

all: build  ## Default target; same as build.

PYTHON ?= .venv/bin/python

build/build.ninja:
	@cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=$(LLVM_BIN)/clang++ \
	-G Ninja -S . -B build

configure: build/build.ninja  ## Run CMake configure only, no compile (build/).

build: build/build.ninja  ## Configure and compile the CLI, library, and tests (build/).
	@cmake --build build -j

build-cuda/build.ninja:
	@cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=$(LLVM_BIN)/clang++ \
	-DBONSAI_CUDA=ON \
	-G Ninja -S . -B build-cuda

build-cuda: build-cuda/build.ninja  ## Configure and compile with the CUDA backend (build-cuda/).
	@cmake --build build-cuda -j

test-cuda: build-cuda $(TOY_SENTINEL)  ## Build the CUDA variant and run ctest against it.
	@ctest --test-dir build-cuda

build-asan/build.ninja:
	@cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_CXX_COMPILER=$(LLVM_BIN)/clang++ \
	-DBONSAI_SANITIZE=ON \
	-G Ninja -S . -B build-asan

build-asan: build-asan/build.ninja  ## Build the ASan + UBSan variant (build-asan/).
	@cmake --build build-asan -j

test-asan: build-asan $(TOY_SENTINEL)  ## Build the ASan + UBSan variant and run ctest (CI-only on macOS).
	@ctest --test-dir build-asan

clean:  ## Remove build/, build-cuda/, and build-asan/.
	@rm -rf build build-cuda build-asan

rebuild: clean build  ## Clean, then build.

format:  ## clang-format in place over src/, include/, tests/, benchmarks/.
	@$(LLVM_BIN)/clang-format -i $(SOURCES)

format-check:  ## Check formatting with clang-format --dry-run --Werror (CI gate).
	@$(LLVM_BIN)/clang-format --dry-run --Werror $(SOURCES)

# Ruff, pinned the way LLVM is: linter drift breaks CI, not the code.
RUFF_VERSION := 0.15.21

lint-python:  ## Run ruff over python/ and scripts/ (pinned via uvx).
	@uvx ruff@$(RUFF_VERSION) check python scripts

# run-clang-tidy exits non-zero when findings exist; a non-zero exit with
# no findings means the tool itself failed and must not pass silently.
lint: build/build.ninja  ## Run clang-tidy over src/, header-filtered to bonsai.
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

run: build  ## Build, then run ./build/src/bonsai with ARGS.
	@./build/src/bonsai $(ARGS)

# Re-extract the parameters reference input from the built CLI, then rerender
# docs/use/parameters.md. `bonsai params` dumps the default Config as TOML
# straight from the structs; the extract step needs Python 3.11+ (tomllib).
# CI runs only render_params.py --check against the committed JSON, so it
# never rebuilds the CLI to verify the page.
params-json: build  ## Re-extract docs/use/parameters.src.json from the built CLI and rerender the page.
	@./build/src/bonsai params | python3 scripts/render_params.py --extract
	@python3 scripts/render_params.py

test: build $(TOY_SENTINEL)  ## Build, fetch the pinned test datasets, run ctest.
	@ctest --test-dir build

# Python extension. Needs a python with nanobind + numpy installed
# (override with PYTHON=/path/to/python); below 3.11 the stub step also needs
# typing_extensions.
python: build/build.ninja  ## Build the _bonsai Python extension into build/python/.
	@cmake -B build -DBONSAI_PYTHON=ON -DBONSAI_OPENMP_STATIC=ON \
	    -DBONSAI_OPENMP_DYNAMIC_FALLBACK_OK=ON \
	    -DPython_EXECUTABLE=$(abspath $(PYTHON)) \
	    | grep -iE "openmp|error" || true
	@cmake --build build --target _bonsai bonsai_stub -j
	@echo "module at build/python/bonsai — use PYTHONPATH=build/python"

python-test: python $(TOY_SENTINEL) $(AMAZON_SENTINEL)  ## Build the extension and run the Python test suites.
	@PYTHONPATH=build/python $(PYTHON) -m pytest python/tests -q

# CUDA-enabled extension in the CUDA tree; cuda_* growers can train.
python-cuda: build-cuda/build.ninja  ## Build the CUDA-enabled Python extension into build-cuda/python/.
	@cmake -B build-cuda -DBONSAI_PYTHON=ON -DBONSAI_OPENMP_STATIC=ON \
	    -DBONSAI_OPENMP_DYNAMIC_FALLBACK_OK=ON \
	    -DPython_EXECUTABLE=$(abspath $(PYTHON)) >/dev/null
	@cmake --build build-cuda --target _bonsai -j
	@echo "module at build-cuda/python/bonsai — use PYTHONPATH=build-cuda/python"

fit-benchmark: build $(TOY_SENTINEL)  ## Compare bonsai against reference libraries on California housing.
	@uv run scripts/compare.py --config configs/california_housing.toml $(ARGS)

# Scaling suite (benchmarks/README.md): synthetic rows/cols/bins/threads
# sweep vs xgboost/lightgbm/catboost. The ladders are bundled specs, one per
# axis: ARGS='run --spec scaling-rows' (also scaling-cols/-bins/-threads).
# Uses the CUDA module tree when present, else the CPU one.
bench-scaling:  ## Run a synthetic scaling spec (ARGS='run --spec scaling-rows').
	@PYTHONPATH=$(if $(wildcard build-cuda/python),build-cuda/python,build/python) \
	    uv run scripts/bench_scaling.py $(ARGS)

# Replicates the iso-volume campaign's bonsai arms on this host (the pod
# campaign runs all six arms via scripts/pod_bench_driver.sh). Rows land in
# the shared results file, distinguished by host name and run label.
bench-iso: python-cuda  ## Run the iso-volume bonsai arms on this host's GPU.
	@PYTHONPATH=build-cuda/python $(PYTHON) -m bonsai.bench run \
	    --spec iso-volume-2026-08 \
	    --variants bonsai_cuda_depthwise,bonsai_cuda_levelwise \
	    --out benchmarks/results/iso-volume-2026-08.jsonl \
	    --run-label iso-volume-2026-08-workrig \
	    --host-name workrig-rtx-pro-6000 $(ARGS)

$(TOY_SENTINEL):
	@uv run scripts/fetch_toy.py
	@touch $@

# test_encoding.py's amazon quality pin needs the stage-1 CSVs (gitignored).
$(AMAZON_SENTINEL):
	@uv run scripts/fetch_amazon.py
	@touch $@

help:  ## List the common make targets.
	@echo "Targets:"
	@grep -hE '^[a-zA-Z0-9_-]+:.*  ## ' $(MAKEFILE_LIST) \
	    | sed 's/:.*  ## /\t/' \
	    | awk -F'\t' '{printf "  make %-18s %s\n", $$1, $$2}'

docs-check:  ## Verify generated docs and lint prose (the five CI doc gates).
	@python3 scripts/render_results.py --check
	@python3 scripts/render_params.py --check
	@python3 scripts/render_make_map.py --check
	@python3 scripts/render_timeline.py --check
	@python3 scripts/docs_lint.py
	@python3 scripts/check_standings.py --decisions

install-hooks:  ## Point core.hooksPath at the versioned hooks (commit-msg format gate).
	@chmod +x scripts/git-hooks/*
	@git config core.hooksPath scripts/git-hooks
	@echo "hooks installed: core.hooksPath = scripts/git-hooks"

skills: $(SKILLS_DIR)/caveman/SKILL.md  ## Install project-local Claude Code skills (currently caveman).

$(SKILLS_DIR)/caveman/SKILL.md:
	@mkdir -p $(@D)
	@echo "Fetching caveman skill -> $@"
	@curl -fsSL $(CAVEMAN_URL) -o $@
	@echo "Done. Restart Claude Code (or trigger a skill rediscovery) to pick it up."

skills-clean:  ## Remove installed project-local skills.
	rm -rf $(SKILLS_DIR)/caveman

.PHONY: configure build build-cuda build-asan clean rebuild format format-check lint lint-python all run params-json test test-cuda test-asan fit-benchmark bench-scaling bench-iso python python-cuda python-test docs-check install-hooks skills skills-clean help
