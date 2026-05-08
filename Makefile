.PHONY: install test lint format check build clean release help

PYTHON ?= python
PIP    ?= pip

help:
	@echo "paretobandits — common dev tasks"
	@echo ""
	@echo "  make install       Install package with dev extras (editable)."
	@echo "  make test          Run pytest with coverage."
	@echo "  make lint          Run ruff check."
	@echo "  make format        Run ruff format."
	@echo "  make check         Lint + test."
	@echo "  make build         Build sdist + wheel into dist/."
	@echo "  make clean         Remove build artifacts and caches."
	@echo "  make release VERSION=x.y.z   Tag a release."

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

# Tag a release. Bumps __version__ and creates a git tag.
# Usage: make release VERSION=0.4.0
release:
	@if [ -z "$(VERSION)" ]; then \
		echo "ERROR: VERSION is required. Usage: make release VERSION=0.4.0"; \
		exit 1; \
	fi
	@echo "Bumping to v$(VERSION) ..."
	@sed -i.bak 's/^__version__ = .*/__version__ = "$(VERSION)"/' paretobandits/__init__.py
	@rm paretobandits/__init__.py.bak
	@git add paretobandits/__init__.py CHANGELOG.md
	@git commit -m "Release v$(VERSION)"
	@git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	@echo ""
	@echo "Created tag v$(VERSION). Push with:"
	@echo "  git push origin main && git push origin v$(VERSION)"
	@echo ""
	@echo "The release workflow will then build and upload to PyPI."
