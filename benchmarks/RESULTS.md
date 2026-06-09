# LIMEN scaling benchmark — simulator vs IBM QPU

- Date: 2026-06-09
- Backend: simulator only (no IBM credentials)
- QAOA: p=1, fixed β=γ=0.1, 1000 shots, optimization_level=1 transpilation
- Validator: 1000 runs, seed 42. κ from Stackelberg co-design (simulation mode, ≤10 iterations).


| Problem | Vars | Depth | Confidence | κ | Sim optimal % | QPU optimal % |
|---|---|---|---|---|---|---|
| trivial-2 | 2 | — | 91.9% | 0.690 | 48.1 | — |
| ring-3 | 3 | — | 90.8% | 0.693 | 51.5 | — |
| ring-4 | 4 | — | 90.5% | 0.691 | 44.7 | — |
| ring-6 | 6 | — | 85.7% | 0.669 | 29.2 | — |
| ring-8 | 8 | — | 82.2% | 0.646 | 19.4 | — |
| ring-10 | 10 | — | 78.0% | 0.630 | 12.8 | — |
| ring-12 | 12 | — | 74.7% | 0.612 | 8.5 | — |
