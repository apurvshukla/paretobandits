# Benchmark: synthetic_high_d

horizon=500, n_seeds=3, K=6, M=2, d=2, shifts=[250]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 718.83 ± 4.59    | 306.67 ± 40.20  | 284.67 ± 61.50    | 1.7       |
| Auer16        | 722.24 ± 0.88    | 294.16 ± 3.38   | 249.17 ± 3.55     | 0.5       |
| ScalarizedUCB | 704.98 ± 7.49    | 434.34 ± 2.58   | 123.67 ± 3.00     | 0.7       |
| RandomPlay    | 732.07 ± 0.22    | 372.43 ± 2.44   | 500.00 ± 0.00     | 0.2       |

**Ranking on PreferenceRegret (lower better):**

1. **ScalarizedUCB** — 704.98
2. **PCBShift** — 718.83 ← PCBShift
3. **Auer16** — 722.24
4. **RandomPlay** — 732.07
