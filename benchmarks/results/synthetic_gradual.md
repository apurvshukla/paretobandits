# Benchmark: synthetic_gradual

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[150, 350]

| algorithm            | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| -------------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift             | 977.07 ± 26.05   | 352.76 ± 25.17  | 234.06 ± 16.85    | 0.4       |
| SukKpotufe20         | 1024.56 ± 0.38   | 429.51 ± 4.94   | 289.06 ± 0.28     | 0.1       |
| Turgay18             | 1005.96 ± 26.49  | 402.31 ± 14.83  | 353.94 ± 25.21    | 0.3       |
| Auer16               | 981.91 ± 9.86    | 326.48 ± 7.51   | 236.78 ± 15.81    | 0.1       |
| ScalarizedUCB        | 946.80 ± 2.80    | 463.08 ± 0.92   | 109.44 ± 2.30     | 0.1       |
| RandomPlay           | 1032.63 ± 0.69   | 394.93 ± 2.54   | 500.00 ± 0.00     | 0.1       |
| ParetoUCB            | 1002.97 ± 3.82   | 347.66 ± 7.34   | 395.33 ± 12.23    | 0.2       |
| AnnealingPareto      | 970.93 ± 7.26    | 339.70 ± 10.48  | 213.56 ± 35.46    | 0.2       |
| StaticBinning        | 854.49 ± 20.01   | 290.94 ± 46.07  | 360.72 ± 12.34    | 0.2       |
| SlidingWindowBinning | 959.63 ± 28.63   | 309.43 ± 21.44  | 355.89 ± 18.38    | 0.4       |
| CUSUMRestart         | 896.40 ± 11.60   | 312.42 ± 21.93  | 359.28 ± 5.62     | 0.2       |
| ATCBinning           | 854.49 ± 20.01   | 290.94 ± 46.07  | 360.72 ± 12.34    | 0.5       |

**Ranking on PreferenceRegret (lower better):**

1. **StaticBinning** — 854.49
2. **ATCBinning** — 854.49
3. **CUSUMRestart** — 896.40
4. **ScalarizedUCB** — 946.80
5. **SlidingWindowBinning** — 959.63
6. **AnnealingPareto** — 970.93
7. **PCBShift** — 977.07 ← PCBShift
8. **Auer16** — 981.91
9. **ParetoUCB** — 1002.97
10. **Turgay18** — 1005.96
11. **SukKpotufe20** — 1024.56
12. **RandomPlay** — 1032.63
