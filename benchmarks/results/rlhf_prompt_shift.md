# Benchmark: rlhf_prompt_shift

horizon=500, n_seeds=3, K=4, M=3, d=4, shifts=[250]

| algorithm     | PreferenceRegret | HausdorffRegret | DominanceCoverage | runtime_s |
| ------------- | ---------------- | --------------- | ----------------- | --------- |
| PCBShift      | 40.24 ± 26.19    | 156.29 ± 45.62  | 288.67 ± 82.14    | 1.4       |
| Auer16        | 66.57 ± 0.00     | 187.67 ± 0.54   | 330.67 ± 1.25     | 0.1       |
| ScalarizedUCB | 47.54 ± 0.00     | 188.77 ± 1.02   | 169.33 ± 0.00     | 0.1       |
| RandomPlay    | 88.86 ± 0.00     | 105.24 ± 0.00   | 500.00 ± 0.00     | 0.1       |

**Ranking on PreferenceRegret (lower better):**

1. **PCBShift** — 40.24 ← PCBShift
2. **ScalarizedUCB** — 47.54
3. **Auer16** — 66.57
4. **RandomPlay** — 88.86
