# Benchmark: fairness_adult

horizon=500, n_seeds=3, K=6, M=2, d=6, shifts=[250]

| algorithm       | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| --------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift        | 648.51 ± 132.91  | 475.51 ± 40.24  | 207.33 ± 182.79   | 1.0       |
| Auer16          | 608.70 ± 199.68  | 463.15 ± 47.93  | 207.67 ± 48.80    | 0.3       |
| ParetoUCB       | 719.24 ± 8.40    | 487.24 ± 8.29   | 357.33 ± 8.98     | 0.2       |
| AnnealingPareto | 749.89 ± 0.00    | 246.69 ± 29.63  | 111.08 ± 34.60    | 0.2       |
| Kone23          | 749.89 ± 0.00    | 324.04 ± 203.65 | 260.00 ± 112.19   | 0.3       |
| Cai24           | 613.43 ± 192.98  | 487.55 ± 9.34   | 275.67 ± 41.12    | 0.9       |
| ScalarizedUCB   | 235.97 ± 215.99  | 302.45 ± 94.50  | 103.33 ± 47.01    | 0.1       |
| RandomPlay      | 749.89 ± 0.00    | 501.58 ± 0.00   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **ScalarizedUCB** — 235.97
2. **Auer16** — 608.70
3. **Cai24** — 613.43
4. **PCBShift** — 648.51 ← PCBShift
5. **ParetoUCB** — 719.24
6. **AnnealingPareto** — 749.89
7. **Kone23** — 749.89
8. **RandomPlay** — 749.89
