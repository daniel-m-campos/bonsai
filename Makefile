CAVEMAN_URL  := https://raw.githubusercontent.com/juliusbrussee/caveman/main/skills/caveman/SKILL.md
SKILLS_DIR   := .claude/skills
LLVM_BIN     := /opt/homebrew/opt/llvm/bin
SOURCES      := $(shell find src include -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) 2>/dev/null)
LINT_SOURCES := $(shell find src -type f -name '*.cpp' 2>/dev/null)

all: build

configure:
	@cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=$(LLVM_BIN)/clang++ \
	-G Ninja -S . -B build

build: configure
	@cmake --build build -j

clean:
	@rm -rf build

rebuild: clean build

format:
	@$(LLVM_BIN)/clang-format -i $(SOURCES)

lint: build
	@out=$$($(LLVM_BIN)/clang-tidy --quiet --extra-arg=--driver-mode=g++ \
	    -header-filter='include/bonsai/.*' -p build $(LINT_SOURCES) 2>/dev/null \
	    | grep -E '(warning|error):' \
	    | grep -v -E '(c\+\+/v1|system-headers|too many)'); \
	if [ -n "$$out" ]; then echo "$$out"; exit 1; fi; \
	echo "lint: no findings."

run: build

test: build
	@ctest --test-dir build

help:
	@echo "Targets:"
	@echo "  make build         Configure + compile."
	@echo "  make rebuild       Clean + build."
	@echo "  make test          Build + run ctest."
	@echo "  make format        clang-format src/ + include/ in place."
	@echo "  make lint          clang-tidy on src/ (header-filtered to bonsai)."
	@echo "  make skills        Install project-local Claude Code skills (currently: caveman)."
	@echo "  make skills-clean  Remove installed project-local skills."

skills: $(SKILLS_DIR)/caveman/SKILL.md

$(SKILLS_DIR)/caveman/SKILL.md:
	@mkdir -p $(@D)
	@echo "Fetching caveman skill -> $@"
	@curl -fsSL $(CAVEMAN_URL) -o $@
	@echo "Done. Restart Claude Code (or trigger a skill rediscovery) to pick it up."

skills-clean:
	rm -rf $(SKILLS_DIR)

.PHONY: configure build clean rebuild format lint all run test skills skills-clean help
