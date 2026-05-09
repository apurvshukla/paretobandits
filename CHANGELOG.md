# Changelog

All notable changes to **paretobandits** will be documented here. Versions follow [SemVer](https://semver.org/) once we hit `1.0`; before that, breaking changes can land in any minor.

## [Unreleased]

## [0.8.0] — 2026-05-08

### Added
- **Two final baselines.** Total now: 14 algorithms (1 hero + 13 baselines).
  - `Kone23` (Kone, Kaufmann & Richert 2023) — Adaptive Pareto Set Identification. Originally a fixed-confidence ID algorithm, not regret minimization. Adapted to streaming by switching to uniform play after the active set stabilizes. Honest about being out-of-design-regime in the docstring; inclusion is informative because it shows PSI algorithms don't transfer to streaming-with-shift settings.
  - `Cai24` (Cai, Cai & Li 2024) — Transfer learning for contextual MABs. Originally a LinUCB-with-transfer-prior algorithm assuming a fixed historical "source" dataset. Simplified port: per-arm shrinkage between recent and all-time empirical means, with the all-time estimate playing the "source" role. Linear-context version on the v0.9 roadmap.

### Notes
- This is "round-trip" coverage of every algorithm cited in Shukla & Kumar 2024 plus all four legacy baselines from the original `script/` code. The benchmark is now feature-complete relative to the paper's literature: 14 algorithms × 11 configs × 5 metrics.
- Both new baselines are explicitly outside their original design regime when run on shift configs. Their docstrings call this out — and their poor performance on shift configs is itself a benchmark result, not a bug.

## [0.7.0] — 2026-05-08

### Added
- **Six additional baselines.** Total now: 11 algorithms (1 hero + 10 baselines).
  - `ParetoUCB` (Drugan & Nowe 2013) — multi-objective UCB with Pareto-optimal candidate set.
  - `AnnealingPareto` (Yahyaa et al. 2014) — temperature-annealed softmax over Pareto-membership scores.
  - `StaticBinning` — fixed M-bin grid baseline; pure UCB with no shift handling.
  - `SlidingWindowBinning` — fixed grid + per-(bin, arm) sliding window for implicit shift handling.
  - `CUSUMRestart` — fixed grid + two-sided CUSUM change-point detection with per-bin restart.
  - `ATCBinning` — fixed grid + Anytime-Tracking-CUSUM (Dey, Garivier, Kaufmann 2025) extended to vector rewards via Bonferroni union over M objectives.
- All 11 algorithms registered in `benchmarks/run.py:ALGORITHMS` and exposed through `paretobandits.algos.legacy`.
- 6 new smoke tests covering the new algorithms; total now 33 tests.

### Notes
- Four algorithms remain on the v0.8+ roadmap and are not yet ported: Slivkins (2011) Contextual Zooming, Kone et al. (2023) Adaptive Pareto Set ID, Yahyaa et al. (2014b) Knowledge-Gradient-MO, and Cai, Cai & Li (2024) Transfer Learning for Contextual MABs. Adding any one is a focused 1–2 week project — the API contract is the same `Algorithm` subclass with `act` / `update` / `pareto_estimate`.

## [0.6.0] — 2026-05-08

### Added
- **RLHF track.** New `RLHFBandit` environment in `paretobandits.envs.rlhf`. Three-objective rewards (helpful, harmless, honest), prompt embedding as context, K response strategies as arms. Two configs: `rlhf_helpful_harmless` (stationary, M=3 ceiling test) and `rlhf_prompt_shift` (production-style training-vs-deployment distribution shift). Runs synthetic-but-realistic by default; supports cached HH-RLHF / PKU-SafeRLHF rewards via `csv_path=` (CSV schema documented in module docstring).
- M=3 reward vectors verified end-to-end through PCBShift, all baselines, all metrics, and the runner. The `n_objectives` is now read from the env's class attribute rather than the YAML, so envs with hard-coded M (like RLHFBandit's M=3) are handled correctly.

## [0.5.0] — 2026-05-08

### Added
- **Fairness track.** New `FairnessBandit` environment in `paretobandits.envs.fairness`. Wraps a tabular fair-classification dataset; per-step rewards are (accuracy_indicator, 1 - |gap|) where the gap is a rolling-window estimate of demographic-parity / equal-opportunity / predictive-parity, configurable via `fairness_metric=`. Three configs: `fairness_adult`, `fairness_compas`, `fairness_german_credit`, each with a different fairness metric and shift schedule. Synthetic-but-realistic generator by default; load real CSVs via `csv_path=`.

### Added (also in this iteration)
- `--scale full|smoke` flag to the benchmark runner: overrides the YAML's horizon and n_seeds at runtime. `full` = (T=10000, n_seeds=20), `smoke` = (T=500, n_seeds=3).
- `--force` flag to re-run configs whose result JSON already exists; default behavior now skips them so an interrupted batch can be resumed safely.
- `make benchmark` (full publication scale, ~40–60 min), `make benchmark-smoke` (~30s for fast iteration), `make benchmark-fresh` (force rerun), `make benchmark-clean` (wipe results).
- Scale-aware `BENCHMARK.md` generator: leads with explicit caveats and a rerun recipe at sandbox scale; writes confident citable language at publication scale.
- Bumped all six canonical synthetic YAML configs to T=10000, n_seeds=20 — the publication-grade defaults. Sandbox-scale runs go through `--scale smoke` rather than YAML edits.

## [0.4.0] — 2026-05-08

### Added
- **Benchmark suite.** Six canonical YAML configs in `benchmarks/configs/`: `synthetic_no_shift`, `synthetic_single_shift`, `synthetic_multi_shift`, `synthetic_gradual`, `synthetic_tree_family`, `synthetic_high_d`. Each pinned to one canonical experimental setup tied to a specific paper claim (Theorem 1, 2, 3, etc).
- `python -m benchmarks.run --config <path.yaml>` to run a single config; `python -m benchmarks.run --all` to run them all and regenerate `BENCHMARK.md`.
- `BENCHMARK.md` — auto-generated leaderboard at the repo root. Currently populated with sandbox-indicative numbers; clearly labeled as needing a publication-grade rerun (T≥10000, n_seeds≥20).
- Per-config detail tables in `benchmarks/results/<config>.md` and machine-readable JSON in `benchmarks/results/<config>.json`.
- `[bench]` optional dependency (`pip install paretobandits[bench]`) — adds PyYAML for config loading.

### Changed
- The "is this benchmark-level?" answer becomes "yes for the infrastructure; the numbers will be once you run at full scale."

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
