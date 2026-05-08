# Changelog

All notable changes to **paretobandits** will be documented here. Versions follow [SemVer](https://semver.org/) once we hit `1.0`; before that, breaking changes can land in any minor.

## [Unreleased]

## [0.3.0] — 2026-05-08

### Added
- `PCBShift` and `SyntheticShift` now accept arbitrary `context_dim`. Standard dyadic branching (2^d children per split) for `d > 1`; legacy `doubling` branching (2^(level+1) children along the single axis) preserved as the default for `d = 1` so v0.2 numbers don't regress.
- `DyadicTree` accepts `context_dim` and `branching` arguments. `DyadicNode` now stores axis-aligned hyper-rectangle bounds as `(d, 2)` arrays and exposes `diameter` and `max_width` properties.
- New tests for `d=2` and `d=3` end-to-end runs, plus dyadic split correctness.

### Changed
- `PCBShift._should_split` uses `leaf.max_width` for the Lipschitz check instead of the legacy 1D `leaf.width`.

## [0.2.0] — 2026-05-08

### Added
- `paretobandits.algos.legacy.SukKpotufe20` — multi-objective adaptation of Suk & Kpotufe (2020), self-tuning bandits over unknown covariate shifts.
- `paretobandits.algos.legacy.Turgay18` — Pareto Contextual Zooming (Türgay, Öner, Tekin 2018). Active-ball partition over joint context-arm space.
- `paretobandits.algos.legacy.Auer16` — Pareto front identification (Auer, Chiang, Ortner, Drugan 2016). Context-free.
- All three baselines exposed at the top-level package.
- Quickstart now compares six algorithms with a ranking on the primary metric.

## [0.1.0] — 2026-05-08

### Added
- Core abstractions: `Algorithm`, `Environment`, `Preference`, `PolyhedralCone`, `PositiveOrthant`, `HalfspaceCone`.
- Hero algorithm: `PCBShift` (Algorithm 1 from Shukla & Kumar, 2024) — adaptive dyadic discretization, pairwise-CI elimination, optimistic-Pareto pruning, Lipschitz-aware splitting. 1D contexts.
- Two trivial baselines: `RandomPlay`, `ScalarizedUCB`.
- Two environments: `SyntheticShift` with five shift schedules (none/single/multi/gradual/tree) and `Warfarin` wrapping the IWPC dose-finding pipeline.
- Five evaluation metrics: `PreferenceRegret` (the paper's d_p), `HausdorffRegret`, `DominanceCoverage`, `ParetoPrecisionRecall`, `RecoveryTime`.
- `Run` / `RunResult` multi-seed driver with .npz save/load.
- `examples/quickstart.py` end-to-end demo.
- `docs/theory_to_code.md` mapping paper notation (β, α, γ, ρ, V_h, t_p, Δ, d_p, Algorithm 1 line numbers, theorems) to code identifiers.
- Smoke test suite.
