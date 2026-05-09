# Benchmark: fairness_compas

horizon=500, n_seeds=3, K=6, M=2, d=8, shifts=[250]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 677.06 ± 73.52   | 498.63 ± 36.54  | 108.33 ± 120.61   | 2.3       |
| Auer16        | 747.93 ± 2.77    | 491.81 ± 18.32  | 263.33 ± 99.67    | 0.3       |
| ScalarizedUCB | 562.46 ± 224.32  | 349.55 ± 120.22 | 19.00 ± 9.20      | 0.1       |
| RandomPlay    | 749.89 ± 0.00    | 535.36 ± 0.00   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **ScalarizedUCB** — 562.46
2. **PCBShift** — 677.06 ← PCBShift
3. **Auer16** — 747.93
4. **RandomPlay** — 749.89
