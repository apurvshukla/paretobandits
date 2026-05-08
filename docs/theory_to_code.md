# Theory ↔ Code Map

This document maps notation from Shukla & Kumar (2024), "Vector preference-based contextual bandits under distributional shifts," to code identifiers in `paretobandits`. If you've read the paper and want to find where a quantity lives in the library — or vice versa — start here.

## Problem setup

| Paper symbol | Meaning | Code |
|---|---|---|
| `K` | Number of arms | `Algorithm.n_arms`, `Environment.n_arms` |
| `M` | Number of objectives | `Algorithm.n_objectives`, `Environment.n_objectives`, `Preference.M` |
| `d` | Context dimension | `Algorithm.context_dim`, `Environment.context_dim` |
| `T` | Time horizon | `Algorithm.horizon`, `Run.horizon` |
| `t_p` | Change point | `Environment.shift_times()` (list) |
| `δ` | Confidence parameter | `Algorithm.delta`, default 0.05 |
| `C` | Preference cone | `core.preference.PolyhedralCone` (with subclasses `PositiveOrthant`, `HalfspaceCone`) |
| `P` | Source distribution | `Environment.context(t)` for `t < tp` |
| `Q` | Target distribution | `Environment.context(t)` for `t >= tp` |
| `μ_k(X)` | Mean reward of arm `k` at context `X` | `Environment.true_means(context)[k]` |
| `r_t` | Observed reward at time `t` | return value of `Environment.step(t, action)` |
| `η_t` | Sub-Gaussian noise | parameterized by `Environment.sigma` |
| `σ` | Noise scale | `Environment.sigma` |
| `Π` | Family of policies | implementations of `Algorithm` |
| `π_t` | Policy at time `t` | `Algorithm.act(context)` (returns sampled arm) |

## Pareto sets and the preference-based metric

| Paper | Meaning | Code |
|---|---|---|
| `P(X)` | True Pareto set at context `X` | `preference.pareto_set(env.true_means(X))` |
| `P_π(X)` | Policy-induced Pareto set | `algo.pareto_estimate(X)` (returns set of arm ids) |
| `Δ(k, P)` | Scale-independent gap (Def. 7) | `preference.gap(point, pareto_points)` (simplified — see note in `preference.py`) |
| `d_p(P_1, P_2)` | Preference-based metric (Def. 8) | computed inside `eval.metrics.PreferenceRegret._d_p` |
| `R(T)` | Preference-based regret (Eq. 2) | `eval.metrics.PreferenceRegret.compute(result, env)` |

## Regularity assumptions

| Paper | Meaning | Code |
|---|---|---|
| `β` | Hölder exponent on the reward function (Assumption 1) | `PCBShift.beta`, `SyntheticShift.beta` |
| `C_β` | Hölder constant | `PCBShift.lipschitz_L`, `SyntheticShift.lipschitz_L` |
| `α, C_α` | Margin condition parameters (Assumption 2) | implicit in environment design — not exposed as a parameter; tune via `SyntheticShift` reward shape |
| `H(β, C_β)` | Hölder function class | the `SyntheticShift` reward model lives in this class by construction |
| `M(α, C_α)` | Margin distribution class | property of `Environment.context()` distribution |
| `Γ(α, C_α, β, C_β, t_p, T)` | Problem class (Def. 9) | jointly defined by `(SyntheticShift parameters, PCBShift parameters)` |
| `D(γ, C_γ)` | Tree-discretized source-target family (Assumption 3) | `SyntheticShift(schedule="tree", ...)` |
| `ρ_h(P, Q)` | Pathak-Ma-Wainwright dissimilarity | not computed explicitly — appears implicitly via the regret bounds |

## Algorithm 1 (PCBShift)

| Paper line | Meaning | Code |
|---|---|---|
| Line 1 (Input) | Tree partition `T` | `utils.tree.DyadicTree`, built in `PCBShift._build_tree` |
| Line 4 (warm-up) | `t < 8K log(KL/δ)` warm-up | `PCBShift.warmup_per_arm * n_arms` (default 2K, configurable) |
| Line 5 | Round-robin warm-up play | `PCBShift.act` first branch |
| Line 8 | Identify bin `(h_t, i_t)` such that `X_t ∈ B(h_t,i_t)` | `tree.find_leaf(x)` |
| Line 9 | Initialise active arms via parent intersection | `DyadicNode.inherit_from(parent)` |
| Line 10 | Estimated Pareto set `P̂(X_t)` | `PCBShift._estimated_pareto(leaf, leaf.active_arms)` |
| Line 11 | Refine active arms by elimination | `PCBShift._eliminate_inferior(leaf, arms)` |
| Line 12 | Play arm uniformly from active set | `PCBShift.act` final return (least-pulled tie-break) |
| Line 13 | Update bin estimates | `PCBShift.update` |
| Line 14 | Split condition (uncertainty < bin width) | `PCBShift._should_split(leaf)` |
| Line 15 | Add children to leaf set | `DyadicTree.split(node)` |
| `μ̂_k,t(h, i)` | Empirical mean (Eq. 4) | `leaf.estimates[k]["mean"]` |
| `n_k,t(h, i)` | Per-arm play count (Eq. 3) | `leaf.estimates[k]["n"]` |
| `ū_k,t(h, i)` | Confidence radius (Eq. 5) | `leaf.estimates[k]["cr"]` (built by `PCBShift._confidence_radius`) |
| `u_k,t(h, i)` | Upper bound (Eq. 6) | `leaf.estimates[k]["ucb"]` |
| `V_h` | Bin width at level `h` | `leaf.width` |
| `L_t` | Leaf set at time `t` | `tree.leaves` |

## Pairwise CI machinery (technical bits)

The implementation uses a tightening over the paper's individual CIs: pairwise CIs that cancel the bin-Lipschitz bias when comparing two arms within the same bin. The relevant code is `PCBShift._pairwise_beta`. This is faithful to the original `script/classes.py:adaptiveBinning` and is what makes the algorithm competitive in finite samples; the paper's regret analysis goes through verbatim either way.

## Theorem-to-experiment map

| Theorem | What to run |
|---|---|
| Theorem 1 (single shift) | `SyntheticShift(schedule="single")` + `PCBShift`, sweep `T` and `K`, plot cumulative `PreferenceRegret` |
| Theorem 2 (`D(γ, C_γ)` family) | `SyntheticShift(schedule="tree")` + `PCBShift`, vary `γ` via `beta` parameter |
| Theorem 3 (multiple shifts) | `SyntheticShift(schedule="multi", shift_times=[t1, t2, ...])` + `PCBShift` |

## Things deliberately *not* matching the paper

A few engineering choices diverge from the paper's notation; none affect the regret bounds:

- The bin's per-arm CI uses `sqrt(2 log(KM/δ) / n)` rather than `sqrt(log(KM/δ) / 2n)` for numerical stability and to match the original code.
- `warmup_per_arm` defaults to 2 (i.e., 2K total warm-up plays) rather than the `8K log(KL/δ)` in Line 4 of Algorithm 1; the latter is asymptotically the same but inflates the constant unnecessarily for typical horizons.
- The Pareto estimate at evaluation time uses *empirical means* (Line 10), not UCBs. The original code experimented with both; mean-based is what matched the paper's intended semantics.
- Arms in both estimated and true Pareto sets contribute zero to `d_p` (the "skip arms in both" trick from `script/classes.py:PCZ.pregret`); see `eval.metrics.PreferenceRegret._d_p` for why.
