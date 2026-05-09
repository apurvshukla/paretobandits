# Benchmark: fairness_german_credit

horizon=500, n_seeds=3, K=6, M=2, d=10, shifts=[125, 250, 375]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 1072.84 ± 69.33  | 229.72 ± 48.59  | 223.83 ± 42.26    | 8.0       |
| Auer16        | 1124.84 ± 0.00   | 315.78 ± 76.45  | 329.50 ± 51.30    | 0.5       |
| ScalarizedUCB | 1124.84 ± 0.00   | 519.93 ± 2.57   | 124.50 ± 18.44    | 0.1       |
| RandomPlay    | 1124.84 ± 0.00   | 124.50 ± 0.00   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **PCBShift** — 1072.84 ← PCBShift
2. **Auer16** — 1124.84
3. **ScalarizedUCB** — 1124.84
4. **RandomPlay** — 1124.84
