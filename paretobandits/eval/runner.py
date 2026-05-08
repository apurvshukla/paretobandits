"""Run an algorithm against an environment over multiple seeds.

The Run class is the only thing users need for the inner loop.  It handles
seed management, per-step bookkeeping (contexts, actions, rewards, Pareto
estimates), and produces a `RunResult` that all metrics consume.

Typical usage:

    env = SyntheticShift(n_arms=10, schedule="single", shift_times=[1500])
    cone = PositiveOrthant(M=2)
    algo = PCBShift(env.n_arms, env.context_dim, env.n_objectives, cone)

    result = Run(env, algo, horizon=3000, n_seeds=20).execute()
    pr = PreferenceRegret(cone)
    summary = pr.summarize(pr.compute(result, env))

The runner re-seeds env and algo for each seed, so multiple seeds are
genuinely independent realizations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.environment import Environment


@dataclass
class RunResult:
    """Container for everything a run produces.

    Shapes (n_seeds = S, horizon = T):
      contexts            : (S, T, context_dim)
      actions             : (S, T)         int
      rewards             : (S, T, M)      float
      pareto_estimates    : list[list[set[int]]]  — outer S, inner T
      shift_times         : list[int]      env-reported change points
      metadata            : dict           freeform (algo name, seeds, etc.)
    """

    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    pareto_estimates: list[list[set[int]]]
    shift_times: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_seeds(self) -> int:
        return self.actions.shape[0]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]

    def save(self, path: str) -> None:
        """Save to .npz. Pareto estimates are serialized as a list-of-lists.

        For Parquet support add `paretobandits[viz]` and use a separate
        helper — keeping the core dependency-light.
        """
        # Convert sets to lists for npz serialization.
        pe = [
            [sorted(list(s)) for s in seed_pe]
            for seed_pe in self.pareto_estimates
        ]
        np.savez_compressed(
            path,
            contexts=self.contexts,
            actions=self.actions,
            rewards=self.rewards,
            pareto_estimates=np.array(pe, dtype=object),
            shift_times=np.array(self.shift_times, dtype=int),
            metadata=np.array([self.metadata], dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> RunResult:
        data = np.load(path, allow_pickle=True)
        pe_raw = data["pareto_estimates"]
        pareto_estimates = [
            [set(s) for s in seed] for seed in pe_raw
        ]
        return cls(
            contexts=data["contexts"],
            actions=data["actions"],
            rewards=data["rewards"],
            pareto_estimates=pareto_estimates,
            shift_times=list(data["shift_times"].tolist()),
            metadata=dict(data["metadata"][0]),
        )


class Run:
    """Sequence-of-seeds runner.

    Args:
        env: an Environment subclass instance.
        algo: an Algorithm subclass instance.  Will be reset() for each seed.
        horizon: number of steps T per seed.
        n_seeds: number of independent runs (default 10).
        seeds: optional explicit seed list of length n_seeds; otherwise
               generated from a base seed.
        base_seed: seed used to derive per-run seeds when `seeds` is None.
        verbose: if True, print a small progress summary per seed.
    """

    def __init__(
        self,
        env: Environment,
        algo: Algorithm,
        horizon: int,
        n_seeds: int = 10,
        seeds: Iterable[int] | None = None,
        base_seed: int = 0,
        verbose: bool = False,
    ):
        if env.n_arms != algo.n_arms:
            raise ValueError(
                f"env.n_arms={env.n_arms} != algo.n_arms={algo.n_arms}"
            )
        if env.n_objectives != algo.n_objectives:
            raise ValueError(
                f"env.n_objectives={env.n_objectives} != algo.n_objectives"
                f"={algo.n_objectives}"
            )
        self.env = env
        self.algo = algo
        self.horizon = int(horizon)
        if seeds is None:
            self.seeds = [base_seed + i for i in range(n_seeds)]
        else:
            self.seeds = list(seeds)
        self.verbose = verbose

    def execute(self) -> RunResult:
        S = len(self.seeds)
        T = self.horizon
        d = self.env.context_dim
        M = self.env.n_objectives

        contexts = np.zeros((S, T, d))
        actions = np.zeros((S, T), dtype=int)
        rewards = np.zeros((S, T, M))
        pareto_estimates: list[list[set[int]]] = []

        for s_idx, seed in enumerate(self.seeds):
            self.env.reset(seed=seed)
            self.algo.reset(seed=seed + 10_000)   # different seed for the algo
            seed_pe: list[set[int]] = []

            for t in range(T):
                ctx = self.env.context(t)
                action = self.algo.act(ctx)
                reward = self.env.step(t, action)
                pareto = self.algo.pareto_estimate(ctx)
                self.algo.update(ctx, action, reward)

                contexts[s_idx, t] = ctx
                actions[s_idx, t] = action
                rewards[s_idx, t] = reward
                seed_pe.append(set(pareto))

            pareto_estimates.append(seed_pe)
            if self.verbose:
                print(f"  seed {seed}: done")

        meta = {
            "algo": self.algo.name,
            "env": type(self.env).__name__,
            "horizon": T,
            "seeds": self.seeds,
        }
        return RunResult(
            contexts=contexts,
            actions=actions,
            rewards=rewards,
            pareto_estimates=pareto_estimates,
            shift_times=self.env.shift_times(),
            metadata=meta,
        )
