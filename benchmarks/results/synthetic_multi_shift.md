# Benchmark: synthetic_multi_shift

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[125, 250, 375]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 1071.67 ± 18.20  | 366.88 ± 15.54  | 230.83 ± 10.10    | 1.8       |
| SukKpotufe20  | 932.52 ± 4.72    | 347.92 ± 4.28   | 363.61 ± 8.00     | 0.6       |
| Turgay18      | 968.62 ± 32.28   | 359.43 ± 21.62  | 390.56 ± 36.70    | 1.3       |
| Auer16        | 925.51 ± 26.26   | 308.64 ± 4.71   | 222.39 ± 3.62     | 0.7       |
| ScalarizedUCB | 965.14 ± 1.72    | 454.56 ± 2.24   | 116.72 ± 2.80     | 0.3       |
| RandomPlay    | 1105.58 ± 0.59   | 387.71 ± 1.78   | 500.00 ± 0.00     | 0.2       |

**Ranking on PreferenceRegret (lower better):**

1. **Auer16** — 925.51
2. **SukKpotufe20** — 932.52
3. **ScalarizedUCB** — 965.14
4. **Turgay18** — 968.62
5. **PCBShift** — 1071.67 ← PCBShift
6. **RandomPlay** — 1105.58
