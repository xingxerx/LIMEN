# Constructive Compilation and Restricted Universality for LIMEN Analog Substrates

**Status:** Engineering-grade results for LIMEN's restricted Hamiltonian class
(diagonal quadratic / Ising forms). Theorems 1, 2, 4, and 5 are elementary and
proved in full below. Theorem 3 rests on the published parity-encoding
(LHZ) architecture and is stated with a proof sketch and citations. **None of
this constitutes a general universality theorem for arbitrary analog
Hamiltonians** — that remains open research, as previously documented in
`architecture.md`. What this document delivers is the constructive, certifiable
core that the Phase 3 specification called for: every LIMEN analog compilation
now emits a provable error bound, and the conditions under which exact
compilation is possible are characterized per substrate.

This document has not been peer reviewed. Treat it as internal engineering
mathematics; an arXiv write-up would be the right next step if external
scrutiny is wanted.

---

## Setting

LIMEN's analog layer compiles Hamiltonians of the form

```
H(h, J) = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j
```

acting on n sites, with all terms diagonal in the computational (Z) basis.
A spin configuration is s ∈ {−1, +1}^n, and the energy of configuration s is

```
E(s) = Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j.
```

A *compilation* maps a target (h, J) to physical device parameters whose
effective Hamiltonian is some (h′, J′) of the same form. Write
Δh = h′ − h, ΔJ = J′ − J, and ΔH = H(h′, J′) − H(h, J).

---

## Theorem 1 — Exact compilation certificate

**Statement.** For any diagonal ΔH as above,

```
‖ΔH‖_op  =  max_{s ∈ {−1,+1}^n} |ΔE(s)|  ≤  Σ_i |Δh_i| + Σ_{i<j} |ΔJ_ij|.
```

The middle quantity is computable exactly in O(2^n · m) time (m = number of
nonzero error terms); the right-hand side (the **L1 bound**) is computable in
O(m) time and is valid for all n.

**Proof.** ΔH is a real linear combination of commuting operators
{Z_i, Z_i Z_j}, all simultaneously diagonal in the computational basis. A
diagonal Hermitian operator's operator norm equals its largest absolute
eigenvalue, and the eigenvalues of ΔH are exactly the values ΔE(s) over the
2^n computational basis states. This gives the equality. The inequality is
the triangle inequality applied term-by-term, using |s_i| = |s_i s_j| = 1. ∎

**Remarks.** (a) The bound is tight whenever the signs of the error terms can
be simultaneously aligned by some configuration (e.g., errors supported on a
star or any bipartite-compatible sign pattern). (b) LIMEN computes the exact
norm for n ≤ 20 by enumeration and falls back to the L1 bound above that.

**Implementation:** `limen/analog/certificate.py` → `certify_ising()`.
Every analog backend (`run_neutral_atom`, `run_photonic`, `run_bec`) attaches
the resulting `CompilationCertificate` to its result.

---

## Theorem 2 — Native realizability conditions, neutral-atom (van der Waals)

The neutral-atom backend realises couplings via the van der Waals interaction
V(r) = C6 / r^6 between Rydberg-dressed atoms, with ZZ coefficient V(r)/4 > 0,
and per-site detunings Δ_i.

**Statement.**

1. *(Linear terms are free.)* For any h and any fixed atom positions, the
   detuning choice
   ```
   Δ_i = −2 h_i − Σ_{j≠i} V(r_ij) / 2
   ```
   realises the linear part of H(h, J) exactly. Linear coefficients impose no
   constraint on realizability.

2. *(Quadratic terms are constrained.)* A target (h, J) is **natively
   realizable** in 2-D if and only if there exist positions {r_i} ⊂ ℝ² with
   C6 / (4 r_ij^6) = J_ij for every coupled pair. Necessary conditions:
   - **Sign:** J_ij > 0 for every coupled pair (V(r) > 0 always).
   - **Geometry:** the prescribed distances d_ij = (C6 / 4 J_ij)^{1/6} must be
     realizable by points in ℝ² (the distance matrix must be 2-Euclidean
     embeddable; in particular it must satisfy all triangle inequalities).

**Proof.** (1) Expanding the Rydberg occupation-number interaction
V_ij n_i n_j with n = (1 − Z)/2 produces the ZZ coefficient V_ij/4, a linear
contribution +V_ij/4 on each of sites i and j (i.e. −V_ij/2 absorbed across the
pair as written in the detuning sum), and a constant. The stated Δ_i cancels
the unwanted linear contributions and sets the residual linear coefficient to
h_i; this is a closed-form assignment, hence exact. (2) The coupling realised
between atoms i and j is a fixed, strictly positive, strictly decreasing
function of their distance alone; solving C6/(4 r^6) = J_ij for r requires
J_ij > 0 and yields a unique required distance, so realizability is exactly
the 2-Euclidean embeddability of the induced distance matrix. ∎

**Consequence.** Native universality *fails* for the vdW neutral-atom path:
any target with a ferromagnetic (negative) coupling, or with a geometrically
frustrated distance matrix, cannot be compiled exactly without encoding
overhead. LIMEN classifies this per-instance
(`certificate.natively_realizable`, currently the sign condition; the
geometric condition is measured a posteriori by the Theorem 1 certificate on
the spring-layout output) and quantifies the native-approximation gap.

---

## Theorem 3 — Universality of the quadratic class via parity (LHZ) encoding

**Statement.** Every target (h, J) with arbitrary signs on n logical spins can
be compiled exactly (in coefficient terms) onto a 2-D neutral-atom array using
K = n(n−1)/2 physical qubits, by the parity / LHZ encoding: each physical
qubit represents the relative parity s_i s_j of one logical pair, logical
couplings J_ij become *local fields* on the parity qubits, and consistency is
enforced by 3- and 4-body constraints on plaquettes of physically adjacent
parity qubits. The overhead is quadratic in n and the resulting interaction
graph is planar with bounded degree, hence geometrically embeddable.

**Proof sketch.** Under the change of variables σ_{ij} = s_i s_j the quadratic
term J_ij s_i s_j becomes the linear term J_ij σ_{ij}; by Theorem 2(1), linear
terms are realised exactly by detunings, eliminating both the sign and the
geometry obstruction. The image of the map s ↦ (s_i s_j) is characterized by
closed-loop parity constraints; LHZ showed that n(n−1)/2 − (n−1) independent
plaquette constraints (3- or 4-body, between geometrically adjacent qubits)
suffice, and that constraint violations cost a tunable energy penalty whose
ground-space coincides with the logical problem for sufficiently large
penalty strength. Constraint realizations on Rydberg arrays use small ancilla
gadgets. Full constructions and penalty-strength bounds:

- W. Lechner, P. Hauke, P. Zoller, *A quantum annealing architecture with
  all-to-all connectivity from local interactions*, Science Advances **1**,
  e1500838 (2015).
- M. Lanthaler, W. Lechner, *Minimal constraint count for the parity
  architecture*, J. Math. Phys. **62**, 042201 (2021).
- Rydberg platform background: M. Saffman, T. G. Walker, K. Mølmer,
  Rev. Mod. Phys. **82**, 2313 (2010). ∎

**Status in LIMEN.** Implemented as an automatic compiler pass
(`limen/analog/lhz.py`, wired into `run_neutral_atom` in
`limen/analog/backends/neutral_atom.py`). When `natively_realizable` is
False, compilation now recurses through `lhz_parity_pass` automatically:
the encoded Hamiltonian's logical couplings become local fields, which are
always natively realizable (Theorem 2 part 1), so the recursive compile
yields a real, certified result (`NeutralAtomResult.lhz_result` /
`lhz_certificate`) instead of leaving the caller with only the flagged
heuristic certificate.

---

## Theorem 4 — Exact GBS encoding with provable device validity (photonic)

The photonic backend uses the Gaussian Boson Sampling encoding of
Arrazola & Bromley, PRL **121**, 030503 (2018): adjacency matrix
A = −J/scale, requiring spectral radius ρ(A) < 1 for A to correspond to a
physical GBS device (Takagi decomposition into squeezers + interferometer).

**Statement.** Let scale = 1.1 · max_i Σ_{j≠i} |J_ij| (1.1 × the Gershgorin
row-sum bound), with scale = 1 when J = 0. Then:

1. ρ(A) ≤ 1/1.1 < 1 for **every** coupling matrix J — the encoding always
   produces a valid GBS device.
2. The encoding is exact and invertible: J = −A · scale recovers the target
   coefficients with zero error, so the Theorem 1 certificate is identically
   zero.
3. Ground-state *inference* from GBS samples remains heuristic: the encoding
   guarantees correctness of the device, not concentration of samples on
   optimal assignments.

**Proof.** (1) For any matrix, ρ(A) ≤ max_i Σ_j |A_ij| (Gershgorin / induced
∞-norm bound). Here Σ_j |A_ij| = (Σ_j |J_ij|)/scale ≤ (max row sum)/(1.1 ·
max row sum) = 1/1.1. (2) Immediate from the definition — the map J ↦ A is
multiplication by a nonzero scalar. (3) is the content of the Arrazola-Bromley
proposal itself. ∎

**Bug fixed by this theorem (v0.4.0).** Prior versions scaled by
1.1 · max|J_ij| (entry-wise), which bounds the *entries* of A but not its
spectral radius: a 3-cycle with equal couplings yields ρ(A) ≈ 1.82, an
invalid device, despite the code claiming ρ < 1 "by construction." The
Gershgorin rule makes the claim true. Regression test:
`tests/test_phase3_completion.py::test_photonic_dense_ring_spectral_radius_below_one`.

---

## Theorem 5 — Coefficient-exact superexchange compilation (BEC)

The BEC backend targets a two-component Bose gas in an optical lattice, deep
in the Mott-insulator regime at unit filling.

**Statement.** Second-order perturbation theory in the tunneling t_ij
produces an effective spin Hamiltonian with Ising coupling
J_ij = 4 t_ij² / U, with sign tunable via state-dependent lattices and
scattering-length ratios (Duan, Demler & Lukin, PRL **91**, 090402 (2003)).
Consequently:

1. *(Coefficient exactness.)* For any target (h, J), the assignment
   t_ij = √(|J_ij| · U) / 2, sign flag σ_ij = sgn(J_ij), and per-site
   potential offsets ε_i = h_i reproduces the target coefficients exactly;
   the Theorem 1 certificate is identically zero.
2. *(Validity domain.)* The mapping is physical only in the perturbative
   regime; LIMEN enforces max_ij t_ij / U ≤ 0.25 and flags violations
   (`mott_regime_valid = False`).
3. *(Geometric restriction.)* Superexchange is a tunneling-mediated
   nearest-neighbour effect, so arbitrary coupling *graphs* require lattice
   embedding — the BEC analogue of Theorem 2's geometry condition. Unlike the
   vdW case there is **no sign obstruction**.

**Proof.** (1) is solving J = 4t²/U for t, exact by construction. (2) and (3)
are the validity conditions of the second-order expansion and of the
underlying lattice geometry, per the cited reference. ∎

---

## What is and is not established

| Claim | Status |
|---|---|
| Every analog compilation carries a provable operator-norm error bound | **Proved** (Thm 1), implemented, tested |
| Exact norm computable for n ≤ 20 | **Proved** (Thm 1), implemented |
| vdW neutral-atom: exact characterization of natively realizable targets | **Proved** (Thm 2); sign condition implemented, geometry measured via certificate |
| Arbitrary-sign quadratic targets compile with quadratic ancilla overhead | **Established in literature** (Thm 3, LHZ); compiler pass implemented (`run_neutral_atom` auto-fallback) |
| Photonic GBS device validity for all inputs | **Proved** (Thm 4), implemented, regression-tested |
| BEC superexchange coefficient exactness + validity domain | **Proved given the DDL effective model** (Thm 5), implemented |
| Universality for general (non-diagonal, time-dependent) analog Hamiltonians | **Open.** Not claimed. |
| GBS sampling concentrates on optimal assignments | **Heuristic** (Arrazola-Bromley); not claimed |

---

## References

0. W. Lechner, P. Hauke, P. Zoller, Science Advances **1**, e1500838 (2015).
1. M. Lanthaler, W. Lechner, J. Math. Phys. **62**, 042201 (2021).
2. M. Saffman, T. G. Walker, K. Mølmer, Rev. Mod. Phys. **82**, 2313 (2010).
3. J. M. Arrazola, T. R. Bromley, Phys. Rev. Lett. **121**, 030503 (2018).
4. L.-M. Duan, E. Demler, M. D. Lukin, Phys. Rev. Lett. **91**, 090402 (2003).
