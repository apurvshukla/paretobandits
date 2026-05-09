# Benchmark: synthetic_no_shift

horizon=800, n_seeds=4, K=10, M=2, d=1, shifts=[]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | ParetoPrecisionRecall | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------------------- | --------- |
| PCBShift      | 0.00 ± 0.00      | 367.55 ± 83.95  | 654.81 ± 109.25   | 608.17 ± 30.26        | 5.5       |
| SukKpotufe20  | 0.00 ± 0.00      | 533.61 ± 3.32   | 749.94 ± 14.85    | 474.07 ± 1.33         | 1.3       |
| Turgay18      | 0.00 ± 0.00      | 652.98 ± 34.80  | 632.96 ± 35.02    | 380.27 ± 35.75        | 4.1       |
| Auer16        | 0.00 ± 0.00      | 326.86 ± 6.75   | 632.44 ± 41.06    | 606.21 ± 9.28         | 2.6       |
| ScalarizedUCB | 0.00 ± 0.00      | 756.69 ± 5.80   | 252.31 ± 0.57     | 369.47 ± 0.92         | 0.5       |
| RandomPlay    | 0.00 ± 0.00      | 685.45 ± 0.54   | 800.00 ± 0.00     | 375.44 ± 0.94         | 0.3       |

**Ranking on PreferenceRegret (lower better):**

1. **PCBShift** — 0.00 ← PCBShift
2. **SukKpotufe20** — 0.00
3. **Turgay18** — 0.00
4. **Auer16** — 0.00
5. **ScalarizedUCB** — 0.00
6. **RandomPlay** — 0.00
