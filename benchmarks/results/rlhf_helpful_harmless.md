# Benchmark: rlhf_helpful_harmless

horizon=500, n_seeds=3, K=4, M=3, d=4, shifts=[]

| algorithm       | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| --------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift        | 0.00 ± 0.00      | 159.67 ± 25.29  | 365.83 ± 35.68    | 0.9       |
| Auer16          | 0.00 ± 0.00      | 111.16 ± 3.93   | 491.33 ± 7.98     | 0.2       |
| ParetoUCB       | 0.00 ± 0.00      | 132.26 ± 1.20   | 481.83 ± 5.95     | 0.2       |
| AnnealingPareto | 0.00 ± 0.00      | 127.91 ± 91.50  | 415.17 ± 113.31   | 0.1       |
| Kone23          | 0.00 ± 0.00      | 102.86 ± 11.60  | 489.83 ± 2.01     | 0.2       |
| Cai24           | 0.00 ± 0.00      | 110.49 ± 2.90   | 497.33 ± 2.09     | 0.6       |
| ScalarizedUCB   | 0.00 ± 0.00      | 240.07 ± 0.21   | 245.17 ± 3.70     | 0.1       |
| RandomPlay      | 0.00 ± 0.00      | 129.78 ± 0.00   | 500.00 ± 0.00     | 0.0       |

**Ranking on PreferenceRegret (lower better):**

1. **PCBShift** — 0.00 ← PCBShift
2. **Auer16** — 0.00
3. **ParetoUCB** — 0.00
4. **AnnealingPareto** — 0.00
5. **Kone23** — 0.00
6. **Cai24** — 0.00
7. **ScalarizedUCB** — 0.00
8. **RandomPlay** — 0.00
