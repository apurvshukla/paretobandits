"""Fixed-grid baselines: StaticBinning, SlidingWindowBinning,
CUSUMRestart, ATCBinning.

Faithful ports of the four baselines from the original `script/` code.
All four use a fixed M=5 partition of the 1D context space and differ
only in their shift-detection / forgetting strategy:

  - StaticBinning        : no forgetting. Pure baseline; everything is
                           pooled across the entire run.
  - SlidingWindowBinning : forget via a sliding window of size W per
                           (bin, arm). Implicit shift handling.
  - CUSUMRestart         : monitor reward residuals with two-sided
                           CUSUM; reset bin estimates when CUSUM
                           exceeds threshold h.
  - ATCBinning           : Anytime Tracking CUSUM (Dey, Garivier,
                           Kaufmann 2025) extended to vector rewards via
                           Bonferroni union bound over M objectives.

The original code mixed environment + algorithm concerns (a bin had a
"true mean" function attached to it). This refactor strips that out:
each algorithm keeps only the empirical estimates and the per-bin
shift-detection state.

All four are 1D-only — they operate on a fixed M=5 partition of [0,1].
Generalizing to higher d follows the same pattern as PCBShift's
`branching="dyadic"` path.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


class _FixedBinBase(Algorithm):
    """Shared M-bin static-grid bookkeeping.

    Each bin keeps Welford-style per-arm (mean, M2, n, sigma).
    Subclasses override `_should_reset(bin_idx, action, reward)` and
    `_arm_estimate(bin_idx, arm)` to plug in their shift handling.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        n_bins: int = 5,
        rng=None,
    ):
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            n_objectives=n_objectives,
            preference=preference,
            delta=delta,
            horizon=horizon,
            rng=rng,
        )
        if context_dim != 1:
            raise NotImplementedError(
                f"{type(self).__name__} currently supports context_dim=1 only."
            )
        self.n_bins = int(n_bins)
        self._bins: dict[int, dict] = {}
        self._reset_all_bins()
        self._last_bin = 0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._reset_all_bins()
        self._last_bin = 0

    def _reset_all_bins(self) -> None:
        self._bins = {b: self._fresh_bin() for b in range(self.n_bins)}

    def _fresh_bin(self) -> dict:
        return {
            "arm": {
                k: {
                    "mean": np.zeros(self.n_objectives),
                    "M2": np.zeros(self.n_objectives),
                    "n": 0,
                    "sigma": np.ones(self.n_objectives),
                }
                for k in range(self.n_arms)
            },
        }

    # ─── 1D bin lookup ─────────────────────────────────────────────

    def _bin_idx(self, x: float) -> int:
        i = int(x * self.n_bins)
        return max(0, min(i, self.n_bins - 1))

    # ─── Default per-bin Welford update ────────────────────────────

    def _update_arm(self, bin_idx: int, action: int, reward: np.ndarray) -> None:
        est = self._bins[bin_idx]["arm"][action]
        est["n"] += 1
        n = est["n"]
        d = reward - est["mean"]
        est["mean"] = est["mean"] + d / n
        est["M2"] = est["M2"] + d * (reward - est["mean"])
        if n >= 2:
            est["sigma"] = np.sqrt(np.maximum(est["M2"] / (n - 1), 1e-6))

    # ─── Default UCB-style estimate ────────────────────────────────

    def _arm_estimate(self, bin_idx: int, action: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, ucb_offset) for an arm in a bin."""
        est = self._bins[bin_idx]["arm"][action]
        n = max(est["n"], 1)
        log_term = np.log(max(self.n_arms * self.n_objectives / self.delta, 2.0))
        cr = est["sigma"] * np.sqrt(2 * log_term / n)
        return est["mean"], cr

    # ─── Algorithm interface (shared) ──────────────────────────────

    def act(self, context: np.ndarray) -> int:
        x = float(np.atleast_1d(context)[0])
        b = self._bin_idx(x)
        self._last_bin = b
        # Round-robin warm-up: pull each arm once globally.
        if self._t < self.n_arms:
            return int(self._t % self.n_arms)
        # UCB-based Pareto selection.
        ucb = np.zeros((self.n_arms, self.n_objectives))
        counts = np.zeros(self.n_arms, dtype=int)
        for k in range(self.n_arms):
            mean, cr = self._arm_estimate(b, k)
            ucb[k] = mean + cr
            counts[k] = self._bins[b]["arm"][k]["n"]
        mask = self.preference.pareto_set(ucb)
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            candidates = np.arange(self.n_arms)
        # Tie-break to least-pulled in this bin so under-explored arms
        # still get data.
        return int(candidates[np.argmin(counts[candidates])])

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        b = self._last_bin
        self._update_arm(b, action, reward)
        self._maybe_reset(b, action, reward)
        self._t += 1

    def _maybe_reset(self, bin_idx: int, action: int, reward: np.ndarray) -> None:
        """Subclass hook for shift handling. Default: no reset."""
        return

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        x = float(np.atleast_1d(context)[0])
        b = self._bin_idx(x)
        means = np.stack([self._bins[b]["arm"][k]["mean"] for k in range(self.n_arms)])
        played = np.array([self._bins[b]["arm"][k]["n"] >= 1 for k in range(self.n_arms)])
        if played.sum() == 0:
            return set(range(self.n_arms))
        active_idxs = np.where(played)[0]
        mask = self.preference.pareto_set(means[active_idxs])
        result = {int(active_idxs[i]) for i in range(len(active_idxs)) if mask[i]}
        return result if result else set(active_idxs.tolist())


# ─── Concrete subclasses ────────────────────────────────────────────


class StaticBinning(_FixedBinBase):
    """Fixed-grid baseline with no shift handling. Pure UCB on M bins."""

    name = "StaticBinning"


class SlidingWindowBinning(_FixedBinBase):
    """Fixed-grid baseline with a sliding window per (bin, arm).

    Args:
        window_size: per-arm-per-bin reward buffer size (default 200).
        Other args: see _FixedBinBase.
    """

    name = "SlidingWindowBinning"

    def __init__(self, *args, window_size: int = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = int(window_size)
        self._windows: dict[tuple[int, int], deque] = {
            (b, k): deque(maxlen=self.window_size)
            for b in range(self.n_bins)
            for k in range(self.n_arms)
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._windows = {
            (b, k): deque(maxlen=self.window_size)
            for b in range(self.n_bins)
            for k in range(self.n_arms)
        }

    def _update_arm(self, bin_idx: int, action: int, reward: np.ndarray) -> None:
        # Append to window; estimate is computed on-demand from the window.
        self._windows[(bin_idx, action)].append(reward.copy())

    def _arm_estimate(self, bin_idx: int, action: int) -> tuple[np.ndarray, np.ndarray]:
        w = self._windows[(bin_idx, action)]
        n = len(w)
        if n == 0:
            return np.zeros(self.n_objectives), np.full(self.n_objectives, np.inf)
        rewards = np.stack(list(w))
        mean = rewards.mean(axis=0)
        if n >= 2:
            std = rewards.std(axis=0, ddof=1)
            log_term = np.log(max(self.n_arms * self.n_objectives / self.delta, 2.0))
            cr = std * np.sqrt(2 * log_term / n)
        else:
            cr = np.full(self.n_objectives, 10.0)
        return mean, cr

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        x = float(np.atleast_1d(context)[0])
        b = self._bin_idx(x)
        means = []
        played = []
        for k in range(self.n_arms):
            w = self._windows[(b, k)]
            if len(w) == 0:
                played.append(False)
                means.append(np.zeros(self.n_objectives))
            else:
                played.append(True)
                means.append(np.stack(list(w)).mean(axis=0))
        means = np.stack(means)
        played = np.array(played)
        if played.sum() == 0:
            return set(range(self.n_arms))
        active_idxs = np.where(played)[0]
        mask = self.preference.pareto_set(means[active_idxs])
        result = {int(active_idxs[i]) for i in range(len(active_idxs)) if mask[i]}
        return result if result else set(active_idxs.tolist())


class CUSUMRestart(_FixedBinBase):
    """Fixed-grid baseline with two-sided CUSUM change detection.

    When the CUSUM statistic exceeds threshold h, the bin's per-arm
    estimates are reset and exploration restarts in that bin.

    Args:
        cusum_h: threshold (higher = less sensitive). Default 8.0.
        cusum_eps: minimum detectable shift magnitude. Default 0.1.
        cusum_warmup: number of samples before the reference mean is set.
                      Default = 3 * n_arms.
    """

    name = "CUSUMRestart"

    def __init__(
        self,
        *args,
        cusum_h: float = 8.0,
        cusum_eps: float = 0.1,
        cusum_warmup: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cusum_h = float(cusum_h)
        self.cusum_eps = float(cusum_eps)
        self._cusum_warmup = (
            cusum_warmup if cusum_warmup is not None else 3 * self.n_arms
        )
        # Per-bin CUSUM state.
        self._cusum: dict[int, dict] = {b: self._fresh_cusum() for b in range(self.n_bins)}

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._cusum = {b: self._fresh_cusum() for b in range(self.n_bins)}

    def _fresh_cusum(self) -> dict:
        return {
            "pos": np.zeros(self.n_objectives),
            "neg": np.zeros(self.n_objectives),
            "ref_mean": None,
            "warmup_rewards": [],
        }

    def _maybe_reset(self, bin_idx: int, action: int, reward: np.ndarray) -> None:
        cs = self._cusum[bin_idx]
        if cs["ref_mean"] is None:
            cs["warmup_rewards"].append(reward.copy())
            if len(cs["warmup_rewards"]) >= self._cusum_warmup:
                cs["ref_mean"] = np.mean(cs["warmup_rewards"], axis=0)
            return
        residual = reward - cs["ref_mean"]
        cs["pos"] = np.maximum(0.0, cs["pos"] + residual - self.cusum_eps)
        cs["neg"] = np.maximum(0.0, cs["neg"] - residual - self.cusum_eps)
        if np.any(cs["pos"] > self.cusum_h) or np.any(cs["neg"] > self.cusum_h):
            self._bins[bin_idx] = self._fresh_bin()
            self._cusum[bin_idx] = self._fresh_cusum()


class ATCBinning(_FixedBinBase):
    """Anytime Tracking CUSUM with Bonferroni union over M objectives.

    Vector-valued extension of Dey, Garivier, Kaufmann (2025).

    Args:
        alpha: false-alarm budget (default 0.05).
        max_ref_points: max reference points kept per bin (default 50).
    """

    name = "ATCBinning"

    def __init__(
        self,
        *args,
        alpha: float = 0.05,
        max_ref_points: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.max_ref_points = int(max_ref_points)
        self._atc: dict[int, dict] = {b: self._fresh_atc() for b in range(self.n_bins)}

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._atc = {b: self._fresh_atc() for b in range(self.n_bins)}

    def _fresh_atc(self) -> dict:
        return {
            "ref_points": [],          # list of dicts
            "local_t": 0,
            "running_mean": np.zeros(self.n_objectives),
            "running_n": 0,
            "centered_cumsum": np.zeros(self.n_objectives),
        }

    def _atc_threshold(self, n: int, alpha_r: float) -> float:
        """γ_d(n, α_r) = sqrt(6 log n + 2 log(d/α_r) + 2 log(π²/3))."""
        if n <= 1:
            return float("inf")
        d = self.n_objectives
        return float(np.sqrt(
            6.0 * np.log(max(n, 2))
            + 2.0 * np.log(max(d / alpha_r, 2.0))
            + 2.0 * np.log(np.pi**2 / 3.0)
        ))

    def _maybe_reset(self, bin_idx: int, action: int, reward: np.ndarray) -> None:
        atc = self._atc[bin_idx]
        atc["local_t"] += 1
        t_local = atc["local_t"]
        atc["running_n"] += 1
        n_total = atc["running_n"]
        delta_mean = reward - atc["running_mean"]
        atc["running_mean"] = atc["running_mean"] + delta_mean / n_total
        residual = reward - atc["running_mean"]
        atc["centered_cumsum"] = atc["centered_cumsum"] + residual

        if t_local <= 5:
            if t_local in (1, 2, 4):
                atc["ref_points"].append({
                    "time": t_local,
                    "cumsum": atc["centered_cumsum"].copy(),
                })
            return

        sigma_est = 1.0  # treat reward noise as sigma=1; tighter would need an estimate
        S_t = atc["centered_cumsum"]
        alarm = False
        for i, ref in enumerate(atc["ref_points"]):
            n = t_local - ref["time"]
            if n < 5:
                continue
            r_idx = i + 1
            alpha_r = self.alpha * 6.0 / (np.pi**2 * (r_idx**2))
            gamma = self._atc_threshold(n, alpha_r)
            stat = np.abs(S_t - ref["cumsum"]) / (sigma_est * np.sqrt(n))
            if np.any(stat > gamma):
                alarm = True
                break

        # Add ref points at powers of 2.
        if t_local & (t_local - 1) == 0:
            ref_entry = {"time": t_local, "cumsum": S_t.copy()}
            if len(atc["ref_points"]) < self.max_ref_points:
                atc["ref_points"].append(ref_entry)
            else:
                atc["ref_points"].pop(0)
                atc["ref_points"].append(ref_entry)

        if alarm:
            self._bins[bin_idx] = self._fresh_bin()
            self._atc[bin_idx] = self._fresh_atc()
