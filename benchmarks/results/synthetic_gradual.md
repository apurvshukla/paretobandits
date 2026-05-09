# Benchmark: synthetic_gradual

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[150, 350]

| algorithm            | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| -------------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift             | 977.07 ± 26.05   | 352.76 ± 25.17  | 234.06 ± 16.85    | 0.5       |
| SukKpotufe20         | 1024.56 ± 0.38   | 429.51 ± 4.94   | 289.06 ± 0.28     | 0.2       |
| Turgay18             | 1005.96 ± 26.49  | 402.31 ± 14.83  | 353.94 ± 25.21    | 0.3       |
| Auer16               | 981.91 ± 9.86    | 326.48 ± 7.51   | 236.78 ± 15.81    | 0.2       |
| ParetoUCB            | 1002.97 ± 3.82   | 347.66 ± 7.34   | 395.33 ± 12.23    | 0.2       |
| AnnealingPareto      | 970.93 ± 7.26    | 339.70 ± 10.48  | 213.56 ± 35.46    | 0.2       |
| StaticBinning        | 854.49 ± 20.01   | 290.94 ± 46.07  | 360.72 ± 12.34    | 0.3       |
| SlidingWindowBinning | 959.63 ± 28.63   | 309.43 ± 21.44  | 355.89 ± 18.38    | 0.5       |
| CUSUMRestart         | 896.40 ± 11.60   | 312.42 ± 21.93  | 359.28 ± 5.62     | 0.2       |
| ATCBinning           | 854.49 ± 20.01   | 290.94 ± 46.07  | 360.72 ± 12.34    | 0.3       |
| Kone23               | 976.91 ± 4.40    | 345.86 ± 7.61   | 203.78 ± 27.86    | 0.2       |
| Cai24                | 981.38 ± 6.71    | 304.67 ± 6.11   | 291.89 ± 4.83     | 0.8       |
| ScalarizedUCB        | 946.80 ± 2.80    | 463.08 ± 0.92   | 109.44 ± 2.30     | 0.1       |
| RandomPlay           | 1032.63 ± 0.69   | 394.93 ± 2.54   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **StaticBinning** — 854.49
2. **ATCBinning** — 854.49
3. **CUSUMRestart** — 896.40
4. **ScalarizedUCB** — 946.80
5. **SlidingWindowBinning** — 959.63
6. **AnnealingPareto** — 970.93
7. **Kone23** — 976.91
8. **PCBShift** — 977.07 ← PCBShift
9. **Cai24** — 981.38
10. **Auer16** — 981.91
11. **ParetoUCB** — 1002.97
12. **Turgay18** — 1005.96
13. **SukKpotufe20** — 1024.56
14. **RandomPlay** — 1032.63
