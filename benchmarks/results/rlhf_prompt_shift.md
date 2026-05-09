# Benchmark: rlhf_prompt_shift

horizon=500, n_seeds=3, K=4, M=3, d=4, shifts=[250]

| algorithm       | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| --------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift        | 40.24 ± 26.19    | 156.29 ± 45.62  | 288.67 ± 82.14    | 1.2       |
| Auer16          | 66.57 ± 0.00     | 187.67 ± 0.54   | 330.67 ± 1.25     | 0.1       |
| ParetoUCB       | 37.20 ± 8.79     | 90.62 ± 5.53    | 448.78 ± 6.53     | 0.1       |
| AnnealingPareto | 51.05 ± 12.29    | 176.34 ± 16.56  | 253.00 ± 63.33    | 0.1       |
| Kone23          | 49.09 ± 1.93     | 187.80 ± 0.36   | 221.33 ± 37.46    | 0.1       |
| Cai24           | 6.39 ± 2.07      | 101.91 ± 1.75   | 401.89 ± 11.16    | 0.6       |
| ScalarizedUCB   | 47.54 ± 0.00     | 188.77 ± 1.02   | 169.33 ± 0.00     | 0.1       |
| RandomPlay      | 88.86 ± 0.00     | 105.24 ± 0.00   | 500.00 ± 0.00     | 0.0       |

**Ranking on PreferenceRegret (lower better):**

1. **Cai24** — 6.39
2. **ParetoUCB** — 37.20
3. **PCBShift** — 40.24 ← PCBShift
4. **ScalarizedUCB** — 47.54
5. **Kone23** — 49.09
6. **AnnealingPareto** — 51.05
7. **Auer16** — 66.57
8. **RandomPlay** — 88.86
