"""RLHF contextual bandit environment.

Reframes RLHF as a multi-objective contextual bandit:

  context = prompt embedding (low-dim projection of the prompt's hidden state)
  K arms  = K response generation strategies (e.g., temperatures, prompts,
            model checkpoints, decoding configs)
  rewards = (helpful, harmless, honest)   — three objectives by default

Distribution shift comes from the prompt distribution drifting between
training and deployment: pre-shift the environment samples mostly from
one prompt cluster (e.g., "factual Q&A"), post-shift mostly from another
(e.g., "creative writing"). Different arm strategies are optimal for
different prompt clusters, so an algorithm that doesn't adapt loses.

Two ways to use the environment:

  - **Synthetic-but-realistic generator** (default; no external deps).
    Prompt embeddings are sampled from a mixture of Gaussians, where each
    component represents a prompt cluster. Per-arm reward profiles vary
    by cluster — some arms are tuned for one cluster, others for another.
    This is the v0.6 default and what runs in CI.

  - **Cached HH-RLHF / PKU-SafeRLHF rewards** (optional).
    Pass `csv_path=` pointing to a CSV with columns:
        prompt_id, embedding_0, ..., embedding_{d-1},
        arm_id, helpful, harmless, honest
    where each (prompt_id, arm_id) is a row giving the reward-model
    scores for that response strategy on that prompt. The user is
    responsible for producing this CSV from a real RLHF dataset; a
    template script is in `examples/rlhf_dataset_prep.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from paretobandits.core.environment import Environment


class RLHFBandit(Environment):
    """Multi-objective contextual bandit over an RLHF prompt distribution.

    Args:
        n_arms: K response strategies (e.g., 4 = greedy / sample-T0.5 /
                sample-T1.0 / sample-T1.5).
        embedding_dim: dimension of the prompt embedding context (default 4).
        n_clusters: number of prompt clusters (Gaussian components) in the
                    synthetic mode. Default 3.
        sigma: noise std-dev added to per-step rewards (default 0.05).
        csv_path: optional CSV with cached RLHF rewards; see module docstring.
        schedule: "none", "single", or "multi" — controls cluster
                  composition over time.
        shift_times: change-point times.
        seed: RNG seed.

    Notes:
        Three objectives by default: helpful, harmless, honest. The
        synthetic generator gives each arm a distinct profile per cluster
        — e.g., the "high-temperature sampling" arm is helpful-but-not-
        harmless on creative-writing prompts but mediocre on factual Q&A.
        This produces a multi-arm Pareto front with cluster-conditional
        composition, exactly the regime where d_p-aware algorithms are
        supposed to outperform scalarized ones.
    """

    n_objectives = 3
    OBJECTIVE_NAMES = ("helpful", "harmless", "honest")

    def __init__(
        self,
        n_arms: int = 4,
        embedding_dim: int = 4,
        n_clusters: int = 3,
        sigma: float = 0.05,
        csv_path: str | None = None,
        schedule: str = "single",
        shift_times: Sequence[int] | None = None,
        seed: int | None = None,
    ):
        super().__init__(rng=seed)
        if schedule not in ("none", "single", "multi"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.n_arms = int(n_arms)
        self.embedding_dim = int(embedding_dim)
        self.n_clusters = int(n_clusters)
        self.context_dim = self.embedding_dim
        self.sigma = float(sigma)
        self.schedule = schedule

        if schedule == "none":
            self._shifts = []
        elif schedule == "single":
            self._shifts = list(shift_times or [5000])
        else:
            if shift_times is None:
                raise ValueError("multi schedule requires shift_times")
            self._shifts = list(shift_times)

        if csv_path is not None:
            self._cached = self._load_csv(csv_path)
        else:
            self._cached = None
            # Synthetic mode: pre-build cluster centers + per-cluster arm
            # reward profiles. Fixed across seeds for reproducibility.
            init_rng = np.random.default_rng(0)
            self._cluster_means = init_rng.uniform(
                0.1, 0.9, size=(self.n_clusters, self.embedding_dim)
            )
            self._cluster_std = 0.08
            # arm_profile[c, k, m] = reward of arm k on cluster c, objective m.
            # Each arm has a "specialty cluster" where it shines, plus a
            # generic baseline. Some arms are helpful-leaning, others
            # harmless-leaning, others honest-leaning.
            ap = init_rng.uniform(0.3, 0.6, size=(self.n_clusters, self.n_arms, 3))
            for k in range(self.n_arms):
                specialty = k % self.n_clusters
                obj_lean = k % 3
                ap[specialty, k, obj_lean] = float(np.clip(
                    init_rng.uniform(0.75, 0.92), 0.0, 1.0
                ))
                # Make at least one *other* objective decent for that arm so
                # the Pareto front has multiple non-dominated arms per cluster.
                other = (obj_lean + 1) % 3
                ap[specialty, k, other] = float(np.clip(
                    init_rng.uniform(0.55, 0.75), 0.0, 1.0
                ))
            self._arm_profile = ap

        self._cached_cluster: int | None = None
        self._cached_embedding: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._cached_cluster = None
        self._cached_embedding = None

    # ─── Synthetic prompt generation ───────────────────────────────

    def _cluster_distribution(self, t: int) -> np.ndarray:
        """Return P(cluster) at time t."""
        K = self.n_clusters
        # Default uniform over clusters.
        if self.schedule == "none" or not self._shifts:
            return np.ones(K) / K
        if self.schedule == "single":
            tp = self._shifts[0]
            if t < tp:
                # Concentrate on cluster 0.
                p = np.full(K, 0.1 / max(K - 1, 1))
                p[0] = 1.0 - 0.1 if K > 1 else 1.0
                return p / p.sum()
            else:
                # Concentrate on cluster (K-1).
                p = np.full(K, 0.1 / max(K - 1, 1))
                p[K - 1] = 1.0 - 0.1 if K > 1 else 1.0
                return p / p.sum()
        # multi: rotate through clusters, one shift per cluster transition.
        phase = sum(1 for ts in self._shifts if t >= ts) % K
        p = np.full(K, 0.1 / max(K - 1, 1))
        p[phase] = 1.0 - 0.1 if K > 1 else 1.0
        return p / p.sum()

    def _sample_prompt(self, t: int) -> tuple[int, np.ndarray]:
        if self._cached is not None:
            # CSV mode: sample a row and read its prompt.
            n = len(self._cached["prompt_ids"])
            i = int(self.rng.integers(0, n))
            return int(self._cached["prompt_ids"][i]), self._cached["embeddings"][i]
        probs = self._cluster_distribution(t)
        cluster = int(self.rng.choice(self.n_clusters, p=probs))
        emb = self._cluster_means[cluster] + self._cluster_std * self.rng.standard_normal(
            self.embedding_dim
        )
        return cluster, np.clip(emb, 0.0, 1.0)

    def _load_csv(self, path: str) -> dict:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Loading cached RLHF CSVs requires pandas. "
                "Install with: pip install paretobandits[warfarin]"
            ) from e
        df = pd.read_csv(Path(path))
        emb_cols = sorted(c for c in df.columns if c.startswith("embedding_"))
        for col in ("prompt_id", "arm_id", "helpful", "harmless", "honest"):
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column")
        # Group rewards by (prompt_id, arm_id) → vector reward.
        prompt_ids = df["prompt_id"].unique()
        embeddings = np.stack(
            [df[df["prompt_id"] == p][emb_cols].iloc[0].to_numpy() for p in prompt_ids]
        )
        n_arms_csv = int(df["arm_id"].max()) + 1
        rewards = np.zeros((len(prompt_ids), n_arms_csv, 3))
        prompt_to_idx = {p: i for i, p in enumerate(prompt_ids)}
        for _, row in df.iterrows():
            i = prompt_to_idx[row["prompt_id"]]
            k = int(row["arm_id"])
            rewards[i, k] = [row["helpful"], row["harmless"], row["honest"]]
        return dict(
            prompt_ids=prompt_ids, embeddings=embeddings, rewards=rewards
        )

    # ─── Environment interface ─────────────────────────────────────

    def context(self, t: int) -> np.ndarray:
        cluster, emb = self._sample_prompt(t)
        self._cached_cluster = cluster
        self._cached_embedding = emb
        return emb

    def true_means(self, context: np.ndarray) -> np.ndarray:
        if self._cached_cluster is None:
            self.context(0)
        if self._cached is not None:
            return self._cached["rewards"][self._cached_cluster]
        return self._arm_profile[self._cached_cluster]

    def step(self, t: int, action: int) -> np.ndarray:
        means = self.true_means(self._cached_embedding)
        return np.clip(
            means[action] + self.sigma * self.rng.standard_normal(3), 0.0, 1.0
        )

    # ─── Shift hooks ───────────────────────────────────────────────

    def is_shifted(self, t: int) -> bool:
        return self.schedule != "none" and bool(self._shifts) and t >= self._shifts[0]

    def shift_times(self) -> list:
        return list(self._shifts)
