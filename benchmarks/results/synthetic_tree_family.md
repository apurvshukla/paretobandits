# Benchmark: synthetic_tree_family

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[250]

| algorithm            | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| -------------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift             | 711.10 ± 29.99   | 341.46 ± 43.48  | 321.67 ± 51.40    | 0.5       |
| SukKpotufe20         | 734.42 ± 4.71    | 363.99 ± 5.61   | 462.22 ± 4.25     | 0.1       |
| Turgay18             | 714.60 ± 29.77   | 396.69 ± 16.13  | 345.28 ± 41.61    | 0.3       |
| Auer16               | 707.97 ± 29.66   | 328.35 ± 10.80  | 376.61 ± 27.72    | 0.3       |
| ScalarizedUCB        | 614.65 ± 7.10    | 488.90 ± 1.40   | 139.39 ± 14.59    | 0.1       |
| RandomPlay           | 739.45 ± 0.48    | 386.89 ± 1.84   | 500.00 ± 0.00     | 0.1       |
| ParetoUCB            | 738.75 ± 0.15    | 366.64 ± 5.33   | 436.39 ± 6.43     | 0.2       |
| AnnealingPareto      | 663.97 ± 56.77   | 336.33 ± 18.40  | 335.17 ± 46.99    | 0.2       |
| StaticBinning        | 687.29 ± 28.22   | 319.59 ± 28.84  | 351.94 ± 30.80    | 0.2       |
| SlidingWindowBinning | 701.14 ± 49.89   | 314.09 ± 24.17  | 343.72 ± 20.04    | 0.5       |
| CUSUMRestart         | 687.95 ± 28.77   | 322.00 ± 30.40  | 340.89 ± 37.54    | 0.2       |
| ATCBinning           | 687.29 ± 28.22   | 319.59 ± 28.84  | 351.94 ± 30.80    | 0.3       |

**Ranking on PreferenceRegret (lower better):**

1. **ScalarizedUCB** — 614.65
2. **AnnealingPareto** — 663.97
3. **StaticBinning** — 687.29
4. **ATCBinning** — 687.29
5. **CUSUMRestart** — 687.95
6. **SlidingWindowBinning** — 701.14
7. **Auer16** — 707.97
8. **PCBShift** — 711.10 ← PCBShift
9. **Turgay18** — 714.60
10. **SukKpotufe20** — 734.42
11. **ParetoUCB** — 738.75
12. **RandomPlay** — 739.45
