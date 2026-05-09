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
from paretobandits.algos.legacy import Auer16, Turgay18
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
    "ScalarizedUCB": ScalarizedUCB,
    "RandomPlay": RandomPlay,
}

ENVIRONMENTS: dict[str, Any] = {
    "SyntheticShift": SyntheticShift,
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

    # Instantiate the cone — for now, positive orthant on M=2.
    n_objectives = config["environment"].get("n_objectives", 2)
    cone = PositiveOrthant(M=n_objectives)

    # Environment is rebuilt per seed inside Run.execute(), but we need
    # a sample instance for shape introspection.
    env_cfg = dict(config["environment"])
    env = build_environment(env_cfg, seed=0)

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
    """Write a top-level BENCHMARK.md aggregating across all configs."""
    lines: list[str] = []
    lines.append("# Benchmark leaderboard")
    lines.append("")
    lines.append(
        "Auto-generated by `python -m benchmarks.run --all`. To reproduce, "
        "rerun the same command on a clean checkout. Per-config detail tables "
        "live in `benchmarks/results/<config>.md`."
    )
    lines.append("")
    lines.append(
        "**Primary metric: `PreferenceRegret` (the paper's d_p, lower better).** "
        "Cumulative sum over the horizon. ± is std across seeds."
    )
    lines.append("")

    # Collect per-config numbers into a single table.
    all_algos = []
    for r in results:
        for a in r["algorithms"]:
            if a not in all_algos:
                all_algos.append(a)

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
        lines.append(f"- **{r['config_name']}** (T={r['horizon']}, "
                     f"n_seeds={r['n_seeds']}): {' < '.join(chips)}")
    lines.append("")

    lines.append("## Configuration scale")
    lines.append("")
    if results:
        first = results[0]
        lines.append(
            f"This run was executed at T={first['horizon']}, n_seeds={first['n_seeds']} "
            "(indicative scale). For publication-grade numbers we recommend rerunning "
            "with T≥10000 and n_seeds≥20 — set those in each YAML and rerun."
        )
    out_path.write_text("\n".join(lines))


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

    results: list[dict] = []
    for cp in config_paths:
        cfg = load_config(cp)
        result = run_one(cfg)
        results.append(result)
        json_path = args.out / f"{result['config_name']}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        write_per_config_md(result, args.out / f"{result['config_name']}.md")

    if args.all:
        write_aggregated_md(results, Path("BENCHMARK.md"))
        print(f"\nWrote BENCHMARK.md with {len(results)} configs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
