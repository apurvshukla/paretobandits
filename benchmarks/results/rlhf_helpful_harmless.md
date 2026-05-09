# Benchmark: rlhf_helpful_harmless

horizon=500, n_seeds=3, K=4, M=3, d=4, shifts=[]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 0.00 ± 0.00      | 159.67 ± 25.29  | 365.83 ± 35.68    | 1.4       |
| Auer16        | 0.00 ± 0.00      | 111.16 ± 3.93   | 491.33 ± 7.98     | 0.2       |
| ScalarizedUCB | 0.00 ± 0.00      | 240.07 ± 0.21   | 245.17 ± 3.70     | 0.1       |
| RandomPlay    | 0.00 ± 0.00      | 129.78 ± 0.00   | 500.00 ± 0.00     | 0.0       |

**Ranking on PreferenceRegret (lower better):**

1. **PCBShift** — 0.00 ← PCBShift
2. **Auer16** — 0.00
3. **ScalarizedUCB** — 0.00
4. **RandomPlay** — 0.00
