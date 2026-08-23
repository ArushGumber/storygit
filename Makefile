# storygit — the commands that regenerate everything.
#
# The point of a Makefile here is not build orchestration, it is that every artifact in the
# repository has a named command that recreates it. A figure or a number nobody can
# regenerate is a figure or a number nobody can check.

VENV := .venv/bin
PY   := $(VENV)/python

.PHONY: help check test coverage lint types frontend e2e docs diagrams numbers eval gallery screenshots all clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

check: lint types test  ## Everything that must be green before a commit

test:  ## Run the test suite (offline)
	$(PY) -m pytest

coverage:  ## Line coverage over src/, to find what is claimed but not exercised
	$(PY) -m pytest --cov=storygit --cov-report=term-missing

lint:  ## Lint and format-check
	$(VENV)/ruff check src tests eval scripts
	$(VENV)/ruff format --check src tests eval scripts

types:  ## Type-check src/ and eval/ under mypy strict
	$(VENV)/mypy

frontend:  ## Type-check, unit-test, and build the interface
	cd frontend && npm run typecheck && npm test && npm run build

e2e: frontend  ## Boot the real server and drive the loop over HTTP
	bash scripts/e2e_smoke.sh

diagrams:  ## Compile every TikZ diagram to PDF + SVG and sync to the frontend
	cd docs/diagrams && ./build.sh

numbers:  ## Recompute the deterministic metrics and regenerate the TeX macros
	$(PY) -m eval.offline
	$(PY) -m eval.texnumbers

docs: diagrams numbers  ## Rebuild the outward-facing document
	cd docs && tectonic presentable.tex

eval:  ## Run the live evaluation (costs provider quota)
	$(PY) -m eval.run --config full

gallery:  ## Re-record the Gallery sessions (--offline for the free six)
	$(PY) -m eval.record_gallery

screenshots: frontend  ## Shoot every tab for the visual audit
	$(PY) scripts/screenshots.py

all: check frontend e2e docs  ## Everything reproducible without spending quota

clean:  ## Remove build artifacts, never results
	rm -rf .pytest_cache .mypy_cache .ruff_cache frontend/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
