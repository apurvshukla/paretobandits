# Benchmark: synthetic_single_shift

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[250]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 717.92 ± 7.45    | 340.30 ± 32.07  | 242.33 ± 30.32    | 2.1       |
| SukKpotufe20  | 726.94 ± 0.67    | 428.28 ± 3.05   | 268.72 ± 5.43     | 0.5       |
| Turgay18      | 694.66 ± 13.07   | 388.30 ± 8.84   | 360.44 ± 21.71    | 0.9       |
| Auer16        | 726.02 ± 0.31    | 311.71 ± 6.26   | 219.72 ± 18.97    | 0.4       |
| ScalarizedUCB | 716.79 ± 12.71   | 457.64 ± 3.04   | 121.83 ± 8.02     | 0.2       |
| RandomPlay    | 735.00 ± 1.38    | 389.77 ± 0.81   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **Turgay18** — 694.66
2. **ScalarizedUCB** — 716.79
3. **PCBShift** — 717.92 ← PCBShift
4. **Auer16** — 726.02
5. **SukKpotufe20** — 726.94
6. **RandomPlay** — 735.00
