# Benchmark: synthetic_tree_family

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[250]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 711.10 ± 29.99   | 341.46 ± 43.48  | 321.67 ± 51.40    | 2.5       |
| SukKpotufe20  | 734.42 ± 4.71    | 363.99 ± 5.61   | 462.22 ± 4.25     | 0.4       |
| Turgay18      | 714.60 ± 29.77   | 396.69 ± 16.13  | 345.28 ± 41.61    | 0.7       |
| Auer16        | 707.97 ± 29.66   | 328.35 ± 10.80  | 376.61 ± 27.72    | 0.7       |
| ScalarizedUCB | 614.65 ± 7.10    | 488.90 ± 1.40   | 139.39 ± 14.59    | 0.2       |
| RandomPlay    | 739.45 ± 0.48    | 386.89 ± 1.84   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **ScalarizedUCB** — 614.65
2. **Auer16** — 707.97
3. **PCBShift** — 711.10 ← PCBShift
4. **Turgay18** — 714.60
5. **SukKpotufe20** — 734.42
6. **RandomPlay** — 739.45
