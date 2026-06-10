# LIMEN scaling benchmark — simulator vs IBM QPU

- Date: 2026-06-09
- Backend: ibm_kingston (Heron R2, 156 qubits)
- QAOA: p=1, fixed β=γ=0.1, 1000 shots, optimization_level=1 transpilation
- Validator: 1000 runs, seed 42. κ from Stackelberg co-design (simulation mode, ≤10 iterations).
- QPU job id: d8kaajbqv2lc73851nq0

| Problem | Vars | Depth | Confidence | κ | Sim optimal % | QPU optimal % |
|---|---|---|---|---|---|---|
| trivial-2 | 2 | 18 | 91.9% | 0.690 | 48.1 | 52.2 |
| ring-3 | 3 | 49 | 90.8% | 0.693 | 51.5 | 48.4 |
| ring-4 | 4 | 66 | 90.5% | 0.691 | 44.7 | 43.5 |
| ring-6 | 6 | 101 | 85.7% | 0.669 | 29.2 | 29.5 |
| ring-8 | 8 | 119 | 82.2% | 0.646 | 19.4 | 18.9 |
| ring-10 | 10 | 158 | 78.0% | 0.630 | 12.8 | 12.5 |
| ring-12 | 12 | 151 | 74.7% | 0.612 | 8.5 | 7.4 |
