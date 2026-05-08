"""Quickstart: PCBShift vs RandomPlay vs ScalarizedUCB on the synthetic shift environment.

Runs all three algorithms on the same environment over multiple seeds and
prints a comparison table across the metric suite.

Usage:
    python examples/quickstart.py

Tweak the constants at the top to scale up/down — the defaults are sized
for a quick smoke run that finishes in seconds.
"""

import sys
import time

from paretobandits import (
    DominanceCoverage,
    HausdorffRegret,
    ParetoPrecisionRecall,
    PCBShift,
    PositiveOrthant,
    PreferenceRegret,
    RandomPlay,
    Run,
    ScalarizedUCB,
    SukKpotufe20,
    SyntheticShift,
)
from paretobandits.algos.legacy import Auer16, Turgay18

# ─── Configuration ───────────────────────────────────────────────────

N_ARMS = 8
HORIZON = 1500
SHIFT_AT = 700
N_SEEDS = 5
DELTA = 0.05


def main() -> int:
    # Set up the environment.  Two objectives, one shift halfway through.
    cone = PositiveOrthant(M=2)
    env = SyntheticShift(
        n_arms=N_ARMS,
        n_objectives=2,
        beta=1.0,
        lipschitz_L=4.0,
        sigma=0.1,
        schedule="single",
        shift_times=[SHIFT_AT],
    )

    # Six algorithms: hero + two trivial baselines + three from prior work.
    common = dict(
        n_arms=env.n_arms,
        context_dim=env.context_dim,
        n_objectives=env.n_objectives,
        preference=cone,
        delta=DELTA,
    )
    algos = {
        "PCBShift": PCBShift(horizon=HORIZON, beta=1.0, lipschitz_L=4.0, **common),
        "SukKpotufe20": SukKpotufe20(horizon=HORIZON, **common),
        "Turgay18": Turgay18(horizon=HORIZON, **common),
        "Auer16": Auer16(horizon=HORIZON, **common),
        "ScalarizedUCB": ScalarizedUCB(**common),
        "RandomPlay": RandomPlay(**common),
    }

    metrics = {
        "preference_regret": PreferenceRegret(cone),
        "hausdorff_regret": HausdorffRegret(cone),
        "dominance_coverage": DominanceCoverage(cone),
        "pareto_f1": ParetoPrecisionRecall(cone),
    }

    print("=" * 78)
    print("paretobandits quickstart")
    print(f"  env=SyntheticShift  K={N_ARMS}  M=2  T={HORIZON}  shift@{SHIFT_AT}")
    print(f"  seeds={N_SEEDS}  delta={DELTA}")
    print("=" * 78)

    summaries = {algo_name: {} for algo_name in algos}
    for algo_name, algo in algos.items():
        print(f"\n[{algo_name}] running...", end=" ", flush=True)
        t0 = time.time()
        runner = Run(env, algo, horizon=HORIZON, n_seeds=N_SEEDS, base_seed=0)
        result = runner.execute()
        dt = time.time() - t0
        print(f"done in {dt:.1f}s")
        for m_name, metric in metrics.items():
            values = metric.compute(result, env)
            summaries[algo_name][m_name] = metric.summarize(values)

    # ─── Comparison table (compact, with ranking on primary metric) ─
    rule = "─" * 110
    print("\n" + rule)
    print(f"{'Metric':<24}" + "".join(f"{a:>14}" for a in algos))
    print(rule)
    for m_name in metrics:
        row = f"{m_name + ' (cum)':<24}"
        for a in algos:
            s = summaries[a][m_name]
            row += f"{s.cumulative_mean:>14.1f}"
        print(row)
        row = f"{'  ± std':<24}"
        for a in algos:
            s = summaries[a][m_name]
            row += f"{s.cumulative_std:>14.2f}"
        print(row)
    print(rule)
    print(
        "Lower-is-better: *_regret. Higher-is-better: dominance_coverage, pareto_f1.\n"
        "Cum = sum over horizon. ± is std across seeds."
    )

    # Ranking on primary metric.
    pr_ranks = sorted(
        algos.keys(),
        key=lambda a: summaries[a]["preference_regret"].cumulative_mean,
    )
    print("\nRanking on preference_regret (paper's d_p, lower is better):")
    for rank, name in enumerate(pr_ranks, 1):
        v = summaries[name]["preference_regret"].cumulative_mean
        marker = " ← hero" if name == "PCBShift" else ""
        print(f"  {rank}. {name:<14} {v:>9.1f}{marker}")

    # ─── Sanity checks ──────────────────────────────────────────────
    pcb_pr = summaries["PCBShift"]["preference_regret"].cumulative_mean
    sk_pr = summaries["SukKpotufe20"]["preference_regret"].cumulative_mean
    tg_pr = summaries["Turgay18"]["preference_regret"].cumulative_mean
    rnd_pr = summaries["RandomPlay"]["preference_regret"].cumulative_mean

    print("\nSanity checks on preference_regret (paper's d_p):")
    print(f"  PCBShift < RandomPlay?     {pcb_pr:.1f} < {rnd_pr:.1f}   "
          f"{'OK' if pcb_pr < rnd_pr else 'FAIL'}")
    print(f"  PCBShift < SukKpotufe20?   {pcb_pr:.1f} < {sk_pr:.1f}   "
          f"{'OK' if pcb_pr < sk_pr else 'FAIL'}")
    print(f"  PCBShift < Turgay18?       {pcb_pr:.1f} < {tg_pr:.1f}   "
          f"{'OK' if pcb_pr < tg_pr else 'FAIL'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
