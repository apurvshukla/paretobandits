# Benchmark: fairness_adult

horizon=500, n_seeds=3, K=6, M=2, d=6, shifts=[250]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 648.51 ± 132.91  | 475.51 ± 40.24  | 207.33 ± 182.79   | 1.3       |
| Auer16        | 608.70 ± 199.68  | 463.15 ± 47.93  | 207.67 ± 48.80    | 0.3       |
| ScalarizedUCB | 235.97 ± 215.99  | 302.45 ± 94.50  | 103.33 ± 47.01    | 0.1       |
| RandomPlay    | 749.89 ± 0.00    | 501.58 ± 0.00   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **ScalarizedUCB** — 235.97
2. **Auer16** — 608.70
3. **PCBShift** — 648.51 ← PCBShift
4. **RandomPlay** — 749.89
