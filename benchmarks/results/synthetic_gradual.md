# Benchmark: synthetic_gradual

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[150, 350]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 977.07 ± 26.05   | 352.76 ± 25.17  | 234.06 ± 16.85    | 1.7       |
| SukKpotufe20  | 1024.56 ± 0.38   | 429.51 ± 4.94   | 289.06 ± 0.28     | 0.5       |
| Turgay18      | 1005.96 ± 26.49  | 402.31 ± 14.83  | 353.94 ± 25.21    | 1.3       |
| Auer16        | 981.91 ± 9.86    | 326.48 ± 7.51   | 236.78 ± 15.81    | 0.6       |
| ScalarizedUCB | 946.80 ± 2.80    | 463.08 ± 0.92   | 109.44 ± 2.30     | 0.2       |
| RandomPlay    | 1032.63 ± 0.69   | 394.93 ± 2.54   | 500.00 ± 0.00     | 0.2       |

**Ranking on PreferenceRegret (lower better):**

1. **ScalarizedUCB** — 946.80
2. **PCBShift** — 977.07 ← PCBShift
3. **Auer16** — 981.91
4. **Turgay18** — 1005.96
5. **SukKpotufe20** — 1024.56
6. **RandomPlay** — 1032.63
