.PHONY: install test lint format check build clean release help \
        benchmark benchmark-smoke benchmark-clean benchmark-fresh

PYTHON ?= python
PIP    ?= pip

help:
	@echo "paretobandits — common dev tasks"
	@echo ""
	@echo "Development:"
	@echo "  make install       Install package with dev extras (editable)."
	@echo "  make test          Run pytest with coverage."
	@echo "  make lint          Run ruff check."
	@echo "  make format        Run ruff format."
	@echo "  make check         Lint + test."
	@echo "  make build         Build sdist + wheel into dist/."
	@echo "  make clean         Remove build artifacts and caches."
	@echo "  make release VERSION=x.y.z   Tag a release."
	@echo ""
	@echo "Benchmark:"
	@echo "  make benchmark         Full publication-grade run (T=10000, 20 seeds; ~40-60min)."
	@echo "  make benchmark-smoke   Fast iteration (T=500, 3 seeds; ~30s)."
	@echo "  make benchmark-fresh   --force: re-run all configs even if results exist."
	@echo "  make benchmark-clean   Wipe benchmarks/results/ before re-running."

install:
	$(PIP) install -e ".[dev,warfarin]"

test:
	pytest --cov=paretobandits --cov-report=term-missing

lint:
	ruff check paretobandits tests examples

format:
	ruff format paretobandits tests examples

check: lint test

build: clean
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf build/ dist/ *.egg-info paretobandits.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

benchmark:
	$(PYTHON) -m benchmarks.run --all --scale full

benchmark-smoke:
	$(PYTHON) -m benchmarks.run --all --scale smoke

benchmark-fresh:
	$(PYTHON) -m benchmarks.run --all --scale full --force

benchmark-clean:
	rm -rf benchmarks/results/*.json benchmarks/results/*.md

# Tag a release. Bumps __version__ if needed and creates a git tag.
# Idempotent: safe to re-run; skips no-op steps cleanly.
# Usage: make release VERSION=0.4.0
release:
	@if [ -z "$(VERSION)" ]; then \
		echo "ERROR: VERSION is required. Usage: make release VERSION=0.4.0"; \
		exit 1; \
	fi
	@echo "Releasing v$(VERSION) ..."
	@sed -i.bak 's/^__version__ = .*/__version__ = "$(VERSION)"/' paretobandits/__init__.py
	@rm paretobandits/__init__.py.bak
	@if ! git diff --quiet HEAD -- paretobandits/__init__.py 2>/dev/null; then \
		git add paretobandits/__init__.py CHANGELOG.md 2>/dev/null || true; \
		git commit -m "Release v$(VERSION)"; \
		echo "  Bumped __version__ and committed."; \
	else \
		echo "  __version__ already at $(VERSION); nothing to commit."; \
	fi
	@if git rev-parse "v$(VERSION)" >/dev/null 2>&1; then \
		echo "  Tag v$(VERSION) already exists; leaving it alone."; \
	else \
		git tag -a "v$(VERSION)" -m "Release v$(VERSION)"; \
		echo "  Created tag v$(VERSION)."; \
	fi
	@echo ""
	@echo "Push with:"
	@echo "  git push origin main && git push origin v$(VERSION)"
	@echo ""
	@echo "The release workflow will then build and upload to PyPI."
