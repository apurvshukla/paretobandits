# Benchmark: synthetic_multi_shift

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[125, 250, 375]

| algorithm            | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| -------------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift             | 1071.67 ± 18.20  | 366.88 ± 15.54  | 230.83 ± 10.10    | 0.4       |
| SukKpotufe20         | 932.52 ± 4.72    | 347.92 ± 4.28   | 363.61 ± 8.00     | 0.1       |
| Turgay18             | 968.62 ± 32.28   | 359.43 ± 21.62  | 390.56 ± 36.70    | 0.3       |
| Auer16               | 925.51 ± 26.26   | 308.64 ± 4.71   | 222.39 ± 3.62     | 0.1       |
| ScalarizedUCB        | 965.14 ± 1.72    | 454.56 ± 2.24   | 116.72 ± 2.80     | 0.1       |
| RandomPlay           | 1105.58 ± 0.59   | 387.71 ± 1.78   | 500.00 ± 0.00     | 0.1       |
| ParetoUCB            | 1071.21 ± 18.79  | 340.59 ± 12.64  | 427.72 ± 8.95     | 0.2       |
| AnnealingPareto      | 953.04 ± 61.59   | 339.59 ± 24.47  | 198.56 ± 10.87    | 0.2       |
| StaticBinning        | 753.20 ± 88.27   | 217.68 ± 26.84  | 431.67 ± 29.28    | 0.2       |
| SlidingWindowBinning | 792.94 ± 34.06   | 200.46 ± 15.20  | 443.44 ± 18.88    | 0.5       |
| CUSUMRestart         | 790.21 ± 75.23   | 234.35 ± 19.97  | 424.50 ± 22.83    | 0.2       |
| ATCBinning           | 753.20 ± 88.27   | 217.68 ± 26.84  | 431.67 ± 29.28    | 0.3       |

**Ranking on PreferenceRegret (lower better):**

1. **StaticBinning** — 753.20
2. **ATCBinning** — 753.20
3. **CUSUMRestart** — 790.21
4. **SlidingWindowBinning** — 792.94
5. **Auer16** — 925.51
6. **SukKpotufe20** — 932.52
7. **AnnealingPareto** — 953.04
8. **ScalarizedUCB** — 965.14
9. **Turgay18** — 968.62
10. **ParetoUCB** — 1071.21
11. **PCBShift** — 1071.67 ← PCBShift
12. **RandomPlay** — 1105.58
