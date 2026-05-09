# Benchmark: synthetic_high_d

horizon=500, n_seeds=3, K=6, M=2, d=2, shifts=[250]

| algorithm       | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| --------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift        | 718.83 ± 4.59    | 306.67 ± 40.20  | 284.67 ± 61.50    | 0.7       |
| Auer16          | 722.24 ± 0.88    | 294.16 ± 3.38   | 249.17 ± 3.55     | 0.2       |
| ParetoUCB       | 709.81 ± 13.46   | 305.16 ± 6.91   | 421.83 ± 21.47    | 0.2       |
| AnnealingPareto | 721.40 ± 0.74    | 304.21 ± 6.11   | 223.17 ± 12.27    | 0.2       |
| Kone23          | 721.32 ± 0.98    | 300.78 ± 9.17   | 244.33 ± 10.63    | 0.2       |
| Cai24           | 723.96 ± 2.02    | 279.57 ± 7.88   | 323.39 ± 26.18    | 0.9       |
| ScalarizedUCB   | 704.98 ± 7.49    | 434.34 ± 2.58   | 123.67 ± 3.00     | 0.1       |
| RandomPlay      | 732.07 ± 0.22    | 372.43 ± 2.44   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **ScalarizedUCB** — 704.98
2. **ParetoUCB** — 709.81
3. **PCBShift** — 718.83 ← PCBShift
4. **Kone23** — 721.32
5. **AnnealingPareto** — 721.40
6. **Auer16** — 722.24
7. **Cai24** — 723.96
8. **RandomPlay** — 732.07
