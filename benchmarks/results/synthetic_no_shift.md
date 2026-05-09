# Benchmark: synthetic_no_shift

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[]

| algorithm            | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| -------------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift             | 0.00 ± 0.00      | 233.45 ± 63.43  | 428.17 ± 25.60    | 0.5       |
| SukKpotufe20         | 0.00 ± 0.00      | 327.54 ± 8.99   | 450.89 ± 5.45     | 0.1       |
| Turgay18             | 0.00 ± 0.00      | 421.98 ± 19.64  | 375.61 ± 10.08    | 0.3       |
| Auer16               | 0.00 ± 0.00      | 202.18 ± 12.62  | 420.94 ± 25.54    | 0.2       |
| ScalarizedUCB        | 0.00 ± 0.00      | 469.97 ± 1.63   | 227.89 ± 3.10     | 0.1       |
| RandomPlay           | 0.00 ± 0.00      | 437.09 ± 2.96   | 500.00 ± 0.00     | 0.1       |
| ParetoUCB            | 0.00 ± 0.00      | 298.81 ± 4.57   | 455.33 ± 12.73    | 0.3       |
| AnnealingPareto      | 0.00 ± 0.00      | 206.66 ± 16.92  | 423.83 ± 31.36    | 0.2       |
| StaticBinning        | 0.00 ± 0.00      | 247.30 ± 30.97  | 413.94 ± 13.19    | 0.2       |
| SlidingWindowBinning | 0.00 ± 0.00      | 184.81 ± 20.28  | 412.50 ± 10.37    | 0.7       |
| CUSUMRestart         | 0.00 ± 0.00      | 266.22 ± 8.10   | 391.67 ± 11.38    | 0.2       |
| ATCBinning           | 0.00 ± 0.00      | 247.30 ± 30.97  | 413.94 ± 13.19    | 0.3       |

**Ranking on PreferenceRegret (lower better):**

1. **PCBShift** — 0.00 ← PCBShift
2. **SukKpotufe20** — 0.00
3. **Turgay18** — 0.00
4. **Auer16** — 0.00
5. **ScalarizedUCB** — 0.00
6. **RandomPlay** — 0.00
7. **ParetoUCB** — 0.00
8. **AnnealingPareto** — 0.00
9. **StaticBinning** — 0.00
10. **SlidingWindowBinning** — 0.00
11. **CUSUMRestart** — 0.00
12. **ATCBinning** — 0.00
