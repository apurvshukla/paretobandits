# Benchmark: fairness_german_credit

horizon=500, n_seeds=3, K=6, M=2, d=10, shifts=[125, 250, 375]

| algorithm       | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| --------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift        | 1072.84 ± 69.33  | 229.72 ± 48.59  | 223.83 ± 42.26    | 6.6       |
| Auer16          | 1124.84 ± 0.00   | 315.78 ± 76.45  | 329.50 ± 51.30    | 0.3       |
| ParetoUCB       | 1124.84 ± 0.00   | 236.07 ± 16.97  | 396.83 ± 14.66    | 0.2       |
| AnnealingPareto | 825.88 ± 422.79  | 294.06 ± 143.89 | 204.50 ± 164.17   | 0.2       |
| Kone23          | 1124.84 ± 0.00   | 141.19 ± 25.28  | 348.33 ± 107.40   | 0.4       |
| Cai24           | 1124.84 ± 0.00   | 240.28 ± 45.03  | 216.67 ± 46.38    | 0.8       |
| ScalarizedUCB   | 1124.84 ± 0.00   | 519.93 ± 2.57   | 124.50 ± 18.44    | 0.1       |
| RandomPlay      | 1124.84 ± 0.00   | 124.50 ± 0.00   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **AnnealingPareto** — 825.88
2. **PCBShift** — 1072.84 ← PCBShift
3. **Auer16** — 1124.84
4. **ParetoUCB** — 1124.84
5. **Kone23** — 1124.84
6. **Cai24** — 1124.84
7. **ScalarizedUCB** — 1124.84
8. **RandomPlay** — 1124.84
