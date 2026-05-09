# Benchmark: synthetic_no_shift

horizon=500, n_seeds=3, K=6, M=2, d=1, shifts=[]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 0.00 ± 0.00      | 233.45 ± 63.43  | 428.17 ± 25.60    | 0.4       |
| SukKpotufe20  | 0.00 ± 0.00      | 327.54 ± 8.99   | 450.89 ± 5.45     | 0.1       |
| Turgay18      | 0.00 ± 0.00      | 421.98 ± 19.64  | 375.61 ± 10.08    | 0.3       |
| Auer16        | 0.00 ± 0.00      | 202.18 ± 12.62  | 420.94 ± 25.54    | 0.2       |
| ScalarizedUCB | 0.00 ± 0.00      | 469.97 ± 1.63   | 227.89 ± 3.10     | 0.1       |
| RandomPlay    | 0.00 ± 0.00      | 437.09 ± 2.96   | 500.00 ± 0.00     | 0.1       |
| Kone23        | 0.00 ± 0.00      | 205.34 ± 17.44  | 431.83 ± 38.57    | 0.2       |
| Cai24         | 0.00 ± 0.00      | 204.56 ± 17.97  | 423.89 ± 30.23    | 1.0       |

**Ranking on PreferenceRegret (lower better):**

1. **PCBShift** — 0.00 ← PCBShift
2. **SukKpotufe20** — 0.00
3. **Turgay18** — 0.00
4. **Auer16** — 0.00
5. **ScalarizedUCB** — 0.00
6. **RandomPlay** — 0.00
7. **Kone23** — 0.00
8. **Cai24** — 0.00
