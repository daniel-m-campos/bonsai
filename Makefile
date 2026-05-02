# Repo setup tasks. Build is CMake; this Makefile is for one-time / dev-env setup.

CAVEMAN_URL := https://raw.githubusercontent.com/juliusbrussee/caveman/main/skills/caveman/SKILL.md
SKILLS_DIR  := .claude/skills

.PHONY: skills skills-clean help

help:
	@echo "Targets:"
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
