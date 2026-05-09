# Benchmark: synthetic_single_shift

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[250]

| algorithm            | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| -------------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift             | 717.92 ± 7.45    | 340.30 ± 32.07  | 242.33 ± 30.32    | 0.4       |
| SukKpotufe20         | 726.94 ± 0.67    | 428.28 ± 3.05   | 268.72 ± 5.43     | 0.2       |
| Turgay18             | 694.66 ± 13.07   | 388.30 ± 8.84   | 360.44 ± 21.71    | 0.3       |
| Auer16               | 726.02 ± 0.31    | 311.71 ± 6.26   | 219.72 ± 18.97    | 0.2       |
| ScalarizedUCB        | 716.79 ± 12.71   | 457.64 ± 3.04   | 121.83 ± 8.02     | 0.1       |
| RandomPlay           | 735.00 ± 1.38    | 389.77 ± 0.81   | 500.00 ± 0.00     | 0.1       |
| ParetoUCB            | 729.88 ± 7.37    | 320.41 ± 5.20   | 431.89 ± 24.79    | 0.2       |
| AnnealingPareto      | 726.51 ± 0.78    | 313.19 ± 7.39   | 227.17 ± 23.59    | 0.2       |
| StaticBinning        | 594.27 ± 86.91   | 239.42 ± 31.65  | 419.22 ± 4.18     | 0.2       |
| SlidingWindowBinning | 544.74 ± 83.20   | 186.76 ± 25.77  | 434.50 ± 8.98     | 0.5       |
| CUSUMRestart         | 600.31 ± 81.23   | 267.05 ± 11.48  | 406.39 ± 1.43     | 0.2       |
| ATCBinning           | 594.27 ± 86.91   | 239.42 ± 31.65  | 419.22 ± 4.18     | 0.3       |

**Ranking on PreferenceRegret (lower better):**

1. **SlidingWindowBinning** — 544.74
2. **StaticBinning** — 594.27
3. **ATCBinning** — 594.27
4. **CUSUMRestart** — 600.31
5. **Turgay18** — 694.66
6. **ScalarizedUCB** — 716.79
7. **PCBShift** — 717.92 ← PCBShift
8. **Auer16** — 726.02
9. **AnnealingPareto** — 726.51
10. **SukKpotufe20** — 726.94
11. **ParetoUCB** — 729.88
12. **RandomPlay** — 735.00
