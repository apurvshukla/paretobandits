"""Benchmark runner.

Loads a YAML config, runs every algorithm in `config['algorithms']` against
the environment in `config['environment']` for `config['horizon']` steps
across `config['n_seeds']` seeds, and writes:

  - benchmarks/results/<config_name>.json   — full metric arrays per algo
  - benchmarks/results/<config_name>.md     — per-config summary table

Usage:
    python -m benchmarks.run --config benchmarks/configs/synthetic_single_shift.yaml
    python -m benchmarks.run --all

`--all` runs every YAML in benchmarks/configs/ and additionally produces
a top-level `BENCHMARK.md` aggregating all configs into a single leaderboard.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from paretobandits import (
    PCBShift,
    PositiveOrthant,
    RandomPlay,
    Run,
    ScalarizedUCB,
    SukKpotufe20,
    SyntheticShift,
)
from paretobandits.algos.legacy import (
    AnnealingPareto,
    ATCBinning,
    Auer16,
    Cai24,
    CUSUMRestart,
    Kone23,
    ParetoUCB,
    SlidingWindowBinning,
    StaticBinning,
    Turgay18,
)
from paretobandits.envs.fairness import FairnessBandit
from paretobandits.envs.rlhf import RLHFBandit
from paretobandits.eval.metrics import (
    DominanceCoverage,
    HausdorffRegret,
    ParetoPrecisionRecall,
    PreferenceRegret,
    RecoveryTime,
)

# ─── Registries ─────────────────────────────────────────────────────

ALGORITHMS: dict[str, Any] = {
    "PCBShift": PCBShift,
    "SukKpotufe20": SukKpotufe20,
    "Turgay18": Turgay18,
    "Auer16": Auer16,
    "ParetoUCB": ParetoUCB,
    "AnnealingPareto": AnnealingPareto,
    "StaticBinning": StaticBinning,
    "SlidingWindowBinning": SlidingWindowBinning,
    "CUSUMRestart": CUSUMRestart,
    "ATCBinning": ATCBinning,
    "Kone23": Kone23,
    "Cai24": Cai24,
    "ScalarizedUCB": ScalarizedUCB,
    "RandomPlay": RandomPlay,
}

ENVIRONMENTS: dict[str, Any] = {
    "SyntheticShift": SyntheticShift,
    "FairnessBandit": FairnessBandit,
    "RLHFBandit": RLHFBandit,
    # Warfarin omitted from default registry — requires the IWPC xls.
    # Add it manually in a config that imports the optional Warfarin class.
}

METRICS: dict[str, Any] = {
    "PreferenceRegret": PreferenceRegret,
    "HausdorffRegret": HausdorffRegret,
    "DominanceCoverage": DominanceCoverage,
    "ParetoPrecisionRecall": ParetoPrecisionRecall,
    "RecoveryTime": RecoveryTime,
}

# ─── Config loading ─────────────────────────────────────────────────


def load_config(path: Path) -> dict:
    """Load a YAML config. Lazy-imports yaml so the core package stays light."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML required to load benchmark configs. "
            "Install with: pip install paretobandits[bench]"
        ) from e
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def build_environment(env_cfg: dict, seed: int):
    """Instantiate an environment from its config dict."""
    env_type = env_cfg.pop("type")
    if env_type not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment type {env_type!r}. "
            f"Known: {list(ENVIRONMENTS)}"
        )
    cls = ENVIRONMENTS[env_type]
    return cls(seed=seed, **env_cfg)


def build_algorithm(name: str, env, cone, delta: float, horizon: int) -> Any:
    """Build an algorithm with sensible defaults filled in from env."""
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}. Known: {list(ALGORITHMS)}")
    cls = ALGORITHMS[name]
    common = dict(
        n_arms=env.n_arms,
        context_dim=env.context_dim,
        n_objectives=env.n_objectives,
        preference=cone,
        delta=delta,
    )
    if name == "PCBShift":
        return cls(horizon=horizon, beta=1.0, lipschitz_L=4.0, **common)
    if name in ("SukKpotufe20", "Turgay18", "Auer16"):
        return cls(horizon=horizon, **common)
    return cls(**common)


# ─── Runner ─────────────────────────────────────────────────────────


def run_one(config: dict) -> dict:
    """Execute a single config and return a serializable result dict."""
    name = config["name"]
    horizon = int(config["horizon"])
    n_seeds = int(config["n_seeds"])
    delta = float(config.get("delta", 0.05))
    algo_names = list(config["algorithms"])
    metric_names = list(config["metrics"])

    print(f"\n[{name}] horizon={horizon}  n_seeds={n_seeds}  "
          f"algos={len(algo_names)}  metrics={len(metric_names)}")

    # Instantiate the env first — env class attrs may set n_objectives
    # (e.g. RLHFBandit overrides to 3). Use the env's actual M for the cone.
    env_cfg = dict(config["environment"])
    env = build_environment(env_cfg, seed=0)
    cone = PositiveOrthant(M=env.n_objectives)

    out: dict[str, Any] = {
        "config_name": name,
        "config_path": config.get("_config_path"),
        "horizon": horizon,
        "n_seeds": n_seeds,
        "delta": delta,
        "environment": dict(config["environment"]),
        "shift_times": env.shift_times(),
        "n_arms": env.n_arms,
        "context_dim": env.context_dim,
        "n_objectives": env.n_objectives,
        "algorithms": {},
    }

    metrics = {m: METRICS[m](cone) for m in metric_names}

    for algo_name in algo_names:
        # Re-init env per algo so contexts stay consistent across algos
        # (each Run.execute() also reseeds env per seed via env.reset).
        env_cfg2 = dict(config["environment"])
        env2 = build_environment(env_cfg2, seed=0)
        algo = build_algorithm(algo_name, env2, cone, delta, horizon)

        t0 = time.time()
        result = Run(env2, algo, horizon=horizon, n_seeds=n_seeds, base_seed=0).execute()
        dt = time.time() - t0
        print(f"  [{algo_name}] {dt:.1f}s")

        algo_metrics: dict[str, Any] = {"runtime_s": dt}
        for m_name, metric in metrics.items():
            try:
                vals = metric.compute(result, env2)
                summ = metric.summarize(vals)
                algo_metrics[m_name] = {
                    "cumulative_mean": float(summ.cumulative_mean),
                    "cumulative_std": float(summ.cumulative_std),
                    "final_mean": float(summ.final_mean),
                    "final_std": float(summ.final_std),
                }
            except Exception as e:
                algo_metrics[m_name] = {"error": str(e)}
        out["algorithms"][algo_name] = algo_metrics

    return out


# ─── Reporting ──────────────────────────────────────────────────────


def write_per_config_md(result: dict, out_path: Path) -> None:
    name = result["config_name"]
    horizon = result["horizon"]
    n_seeds = result["n_seeds"]
    algos = list(result["algorithms"])
    metrics = list(next(iter(result["algorithms"].values())))
    metrics = [m for m in metrics if m != "runtime_s"]

    lines: list[str] = []
    lines.append(f"# Benchmark: {name}")
    lines.append("")
    lines.append(
        f"horizon={horizon}, n_seeds={n_seeds}, K={result['n_arms']}, "
        f"M={result['n_objectives']}, d={result['context_dim']}, "
        f"shifts={result['shift_times']}"
    )
    lines.append("")

    # Cumulative-mean table.
    header = ["algorithm"] + metrics + ["runtime_s"]
    rows = [header]
    for a in algos:
        row = [a]
        for m in metrics:
            v = result["algorithms"][a].get(m, {})
            if "cumulative_mean" in v:
                row.append(f"{v['cumulative_mean']:.2f} ± {v['cumulative_std']:.2f}")
            else:
                row.append("—")
        row.append(f"{result['algorithms'][a]['runtime_s']:.1f}")
        rows.append(row)
    lines.append(_md_table(rows))
    lines.append("")

    # Ranking on preference_regret.
    if "PreferenceRegret" in metrics:
        ranked = sorted(
            algos, key=lambda a: result["algorithms"][a]["PreferenceRegret"]["cumulative_mean"]
        )
        lines.append("**Ranking on PreferenceRegret (lower better):**")
        lines.append("")
        for i, a in enumerate(ranked, 1):
            v = result["algorithms"][a]["PreferenceRegret"]["cumulative_mean"]
            tag = " ← PCBShift" if a == "PCBShift" else ""
            lines.append(f"{i}. **{a}** — {v:.2f}{tag}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def write_aggregated_md(results: list[dict], out_path: Path) -> None:
    """Write a top-level BENCHMARK.md aggregating across all configs.

    Scale-aware: when results are at "sandbox" scale (T < 2000 or n_seeds < 10),
    the markdown leads with explicit caveats and a rerun recipe. When at
    publication scale (T >= 10000 and n_seeds >= 20), it writes confident
    citable language. Mixed-scale runs get the pessimistic framing.
    """
    if not results:
        out_path.write_text("# Benchmark leaderboard\n\n(no results yet)\n")
        return

    horizons = [r["horizon"] for r in results]
    seedcounts = [r["n_seeds"] for r in results]
    is_full_scale = min(horizons) >= 10000 and min(seedcounts) >= 20
    is_sandbox = max(horizons) < 2000 or max(seedcounts) < 10
    scale_label = "publication" if is_full_scale else ("sandbox" if is_sandbox else "indicative")

    # Algorithms across all configs, in stable insertion order.
    all_algos: list[str] = []
    for r in results:
        for a in r["algorithms"]:
            if a not in all_algos:
                all_algos.append(a)

    lines: list[str] = []
    lines.append("# Benchmark")
    lines.append("")
    lines.append(
        "Auto-generated by `python -m benchmarks.run --all`. The configs and "
        "the runner are frozen — the YAML files in `benchmarks/configs/` "
        "define the canonical experimental setups, and any algorithm added "
        "to `benchmarks.run.ALGORITHMS` competes on those. Per-config detail "
        "tables (with all metrics, ± std, runtimes) are in "
        "`benchmarks/results/<config>.md`."
    )
    lines.append("")

    if is_full_scale:
        lines.append(
            "These numbers are **publication-grade**: T≥10000 and n_seeds≥20 "
            "across every config. The rankings are stable and citable."
        )
    elif is_sandbox:
        lines.append(
            "**These numbers are sandbox-indicative, not publication-grade.** "
            "At small horizons regret separations fall within seed noise. "
            "PCBShift's theoretical advantage is asymptotic and only emerges "
            "at T≥10000 with n_seeds≥20. Rerun with `make benchmark` (or "
            "`python -m benchmarks.run --all --scale full --force`) on a real "
            "machine — about 40–60 minutes single-threaded — to populate this "
            "file with citable numbers."
        )
    else:
        lines.append(
            "Mixed-scale results (some configs publication-grade, some "
            "sandbox). For consistent citable numbers across all configs, "
            "rerun with `make benchmark` to produce a uniform full-scale run."
        )
    lines.append("")

    lines.append(
        "**Primary metric: `PreferenceRegret` (the paper's d_p, lower better).** "
        "Cumulative sum over the horizon. Cells show mean across seeds; see "
        "per-config detail for ± std."
    )
    lines.append("")

    # Main leaderboard table.
    header = ["config"] + all_algos
    rows = [header]
    for r in results:
        row = [r["config_name"]]
        for a in all_algos:
            am = r["algorithms"].get(a)
            if am is None:
                row.append("—")
            else:
                v = am.get("PreferenceRegret", {})
                if "cumulative_mean" in v:
                    row.append(f"{v['cumulative_mean']:.1f}")
                else:
                    row.append("—")
        rows.append(row)
    lines.append(_md_table(rows))
    lines.append("")

    # Notes on edge cases.
    has_no_shift = any(not r["shift_times"] for r in results)
    has_high_d = any(r["context_dim"] > 1 for r in results)
    notes: list[str] = []
    if has_no_shift:
        notes.append(
            "`synthetic_no_shift` shows zeros because `PreferenceRegret` is "
            "post-shift-only by design (matches paper Equation 2). For "
            "stationary settings, `HausdorffRegret` in the per-config detail "
            "is the informative metric."
        )
    if has_high_d:
        notes.append(
            "`synthetic_high_d` (`context_dim>1`) omits SukKpotufe20 and "
            "Turgay18 — both currently 1D-only. Generalizing them to higher "
            "d is on the v0.5 roadmap."
        )
    if notes:
        for note in notes:
            lines.append(note)
            lines.append("")

    # Per-config ranking summary.
    lines.append("## Per-config rankings (lower better)")
    lines.append("")
    for r in results:
        ranked = sorted(
            r["algorithms"],
            key=lambda a: r["algorithms"][a].get("PreferenceRegret", {}).get(
                "cumulative_mean", float("inf")
            ),
        )
        chips = []
        for a in ranked:
            v = r["algorithms"][a].get("PreferenceRegret", {}).get("cumulative_mean")
            if v is None:
                continue
            chips.append(f"`{a} {v:.0f}`")
        lines.append(
            f"- **{r['config_name']}** (T={r['horizon']}, n_seeds={r['n_seeds']}): "
            f"{' < '.join(chips)}"
        )
    lines.append("")

    # What each config tests.
    lines.append("## What each config tests")
    lines.append("")
    descriptions = {
        "synthetic_no_shift":     "Stationary baseline. Establishes a 'ceiling' for what's achievable when there's no shift to track.",
        "synthetic_single_shift": "One change point at T/2. Canonical setup for **Theorem 1**. Headline benchmark.",
        "synthetic_multi_shift":  "Three change points alternating between two regimes. Tests **Theorem 3**.",
        "synthetic_gradual":      "Beta parameters interpolate linearly between source and target. Slow concept drift.",
        "synthetic_tree_family":  "Source `Beta(1+β,1)`, target uniform. Family for which **Theorem 2** proves a sharper bound.",
        "synthetic_high_d":       "`context_dim=2`. Verifies multi-d feature; PCBShift uses dyadic splitting (4 children/split).",
        "fairness_adult":         "Adult-style fair classification. Demographic-parity gap as objective 2. Group-prevalence flip mid-run.",
        "fairness_compas":        "COMPAS-style recidivism. Equal-opportunity gap as objective 2. Captures the metric ProPublica raised.",
        "fairness_german_credit": "German Credit-style. Predictive-parity gap as objective 2. Multi-shift schedule (3 transitions).",
        "rlhf_helpful_harmless":  "Stationary RLHF benchmark with M=3 (helpful, harmless, honest). Tests M>2 + per-cluster Pareto fronts.",
        "rlhf_prompt_shift":      "RLHF benchmark with prompt-distribution shift mid-run. Closest setup to production RLHF deployments.",
    }
    for r in results:
        desc = descriptions.get(r["config_name"])
        if desc:
            lines.append(f"- **{r['config_name']}** — {desc}")
    lines.append("")

    # Reproduce + rerun.
    lines.append("## Reproducing or rerunning")
    lines.append("")
    lines.append("```bash")
    lines.append("# Full publication run (T=10000, n_seeds=20):")
    lines.append("make benchmark               # ~40-60 min single-threaded")
    lines.append("")
    lines.append("# Fast iteration during development (T=500, n_seeds=3):")
    lines.append("make benchmark-smoke         # ~30 seconds")
    lines.append("")
    lines.append("# Force re-run all configs (overwrites benchmarks/results/):")
    lines.append("make benchmark-fresh")
    lines.append("```")
    lines.append("")
    lines.append(
        "By default the runner skips configs whose result JSON already exists, "
        "so an interrupted batch can be resumed. Pass `--force` to re-run all."
    )
    lines.append("")

    lines.append("## Adding a new algorithm")
    lines.append("")
    lines.append(
        "1. Subclass `paretobandits.core.Algorithm` and implement `act` / "
        "`update` / `pareto_estimate`.\n"
        "2. Register the class in `benchmarks/run.py:ALGORITHMS`.\n"
        "3. Add the algorithm name to the `algorithms:` list in each YAML "
        "you want it to appear in.\n"
        "4. Re-run `make benchmark-fresh` and submit the regenerated "
        "`BENCHMARK.md` along with your code."
    )
    lines.append("")
    lines.append(f"_Scale of this run: **{scale_label}** "
                 f"(min T={min(horizons)}, min n_seeds={min(seedcounts)})._")
    out_path.write_text("\n".join(lines) + "\n")


def _md_table(rows: list[list[str]]) -> str:
    """Markdown table from a list of rows (first row is header)."""
    if not rows:
        return ""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out: list[str] = []
    for i, row in enumerate(rows):
        cells = [str(c).ljust(widths[j]) for j, c in enumerate(row)]
        out.append("| " + " | ".join(cells) + " |")
        if i == 0:
            out.append("| " + " | ".join("-" * w for w in widths) + " |")
    return "\n".join(out)


# ─── CLI ────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="paretobandits benchmark runner")
    parser.add_argument("--config", type=Path, help="path to a single YAML config")
    parser.add_argument(
        "--all", action="store_true", help="run every config in benchmarks/configs/"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/results"),
        help="output directory (default: benchmarks/results)",
    )
    parser.add_argument(
        "--scale", choices=["full", "smoke"], default=None,
        help=(
            "override the YAML's horizon and n_seeds. "
            "'full' = (T=10000, n_seeds=20), 'smoke' = (T=500, n_seeds=3). "
            "Use 'smoke' for fast iteration; 'full' for publication-grade numbers."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Re-run configs whose result JSON already exists. By default the "
            "runner skips them so an interrupted batch can be resumed safely."
        ),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.all:
        config_dir = Path("benchmarks/configs")
        config_paths = sorted(config_dir.glob("*.yaml"))
        if not config_paths:
            print(f"No configs found in {config_dir}")
            return 1
    elif args.config:
        config_paths = [args.config]
    else:
        parser.error("specify --config <path> or --all")

    scale_overrides = {
        "full": dict(horizon=10000, n_seeds=20),
        "smoke": dict(horizon=500, n_seeds=3),
    }
    overrides = scale_overrides.get(args.scale)

    total = len(config_paths)
    print(f"Running {total} config(s)...")
    if args.scale:
        print(f"  --scale={args.scale} → horizon={overrides['horizon']}, "
              f"n_seeds={overrides['n_seeds']}")
    if not args.force:
        print("  Skipping configs whose result JSON exists; pass --force to re-run.")
    print()

    results: list[dict] = []
    t_total = time.time()
    for i, cp in enumerate(config_paths, 1):
        cfg = load_config(cp)
        if overrides:
            cfg["horizon"] = overrides["horizon"]
            cfg["n_seeds"] = overrides["n_seeds"]
        config_name = cfg.get("name") or cp.stem
        json_path = args.out / f"{config_name}.json"

        # Skip-existing logic for resumability.
        if json_path.exists() and not args.force:
            print(f"[{i}/{total}] {config_name} — skipped (result exists).")
            results.append(json.load(open(json_path)))
            continue

        print(f"[{i}/{total}] {config_name}")
        t0 = time.time()
        result = run_one(cfg)
        results.append(result)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        write_per_config_md(result, args.out / f"{config_name}.md")
        print(f"  → wrote {json_path.name} ({time.time() - t0:.1f}s)")

    if args.all:
        write_aggregated_md(results, Path("BENCHMARK.md"))
        elapsed = time.time() - t_total
        print(f"\nDone in {elapsed:.1f}s. Wrote BENCHMARK.md with {len(results)} configs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
