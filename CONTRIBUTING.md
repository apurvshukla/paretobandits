# Contributing to paretobandits

Thanks for your interest in contributing. This is a small library and the bar for changes is "does it make multi-objective contextual bandits under shift easier to study and benchmark." If yes, open a PR.

## Quick start

```bash
git clone https://github.com/apurvshukla/paretobandits.git
cd paretobandits
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,warfarin]"
pytest                     # run the test suite
ruff check paretobandits   # lint
python examples/quickstart.py
```

A `Makefile` provides shortcuts: `make test`, `make lint`, `make build`, `make clean`, `make check` (lint + test).

## What's welcome

**New algorithms.** Subclass `paretobandits.core.Algorithm`, implement `act` / `update` / `pareto_estimate`, drop the file under `paretobandits/algos/` (or `paretobandits/algos/legacy/` for ports of prior work). Add at least one smoke test in `tests/` that runs the algorithm end-to-end.

**New environments.** Subclass `paretobandits.core.Environment`. Required: `context(t)`, `step(t, action)`, `true_means(context)`. Optional: `is_shifted(t)`, `shift_times()`. Add tests verifying shape contracts.

**New metrics.** Implement `compute(result, env) -> np.ndarray` of shape `(n_seeds, T)` and `summarize(values) -> MetricSummary`. Add to `paretobandits/eval/metrics.py`.

**Bug fixes** in any of the above are always welcome.

**Documentation, especially `docs/theory_to_code.md`** — keeping the paper-to-code mapping current is high leverage.

## What's out of scope (for now)

- Major API refactors of `Algorithm` / `Environment` / `Preference`. The current API is what makes the library composable; please open an issue first to discuss.
- Algorithms that depend on heavy external frameworks (PyTorch, JAX) without a numpy-only fallback.

## Style

- Format with `ruff format` (project uses ruff for both linting and formatting).
- Type hints encouraged but not required across the board; required for public API surfaces.
- Keep modules under ~500 lines. If a baseline is bigger, split into a sub-package.

## Tests

- Every algorithm must have at least one smoke test that runs it end-to-end against `SyntheticShift`.
- Don't ship a regression in 1D `PCBShift` numbers without an explicit reason — the v0.2/v0.3 ranking is the trust anchor.
- Slow benchmark sweeps belong in `benchmarks/`, not `tests/`.

## Releases

Releases are cut by maintainers via `make release VERSION=x.y.z` which:
1. Updates `paretobandits/__init__.py:__version__` and `CHANGELOG.md`.
2. Tags the commit `vx.y.z`.
3. Pushing the tag triggers `.github/workflows/release.yml`, which builds, uploads to TestPyPI, then PyPI, then creates a GitHub release.

PyPI uploads use OIDC trusted publishing — no API tokens in the repo. See `SHIPPING.md` for the one-time PyPI configuration.

## Questions

Open an issue, or tag the maintainers in your PR.
