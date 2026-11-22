# cancel_adjacent: a compiler pass with machine-checked proofs

LIMEN's `CancelInverses` pass removes pairs of gates that undo each other:
H H, X X, Y Y, Z Z, CX CX and SWAP SWAP on the same qubits. The loop that does
the work is `cancel_adjacent` in `rust/limen-pass/src/lib.rs`.

This folder holds Lean 4 proofs about that loop.

**Status, November 2026: work in progress.** Four claims are proven about a
model of the loop. The link from the model to the Rust code is still open.
Both are spelled out below.

## What is proven

Four claims, checked by Lean, about a model of the loop
(`lean/CancelAdjacent/Verification/Spec.lean`):

0. **Never grows.** The output is never longer than the input.
1. **Only removes.** The output is the input with some gates taken out.
   Nothing is added and nothing changes order.
2. **Idempotent.** Running the pass twice gives the same result as running it once.
3. **Meaning preserved,** on one condition: each pair the pass may cancel does
   nothing when run back to back.

## What is not proven yet

- **That the Rust function and the model are the same thing.** The Rust is
  translated to Lean (`lean/CancelAdjacent/Extraction/`) and it compiles, but
  the theorem linking the two is not written. Until it is, the link rests on
  two things: the model follows the Rust loop line for line, and a property
  test in the crate compares the new function with the original loop on
  random circuits.
- **That real gates meet the condition in claim 3.** For example, that H
  followed by H is the identity. That needs matrix math from Mathlib.

The pass makes circuits shorter, not shortest. It only cancels gates that sit
next to each other in the list.

## Reading one of the theorems

```lean
theorem cancel_idempotent (c : α → α → Bool) (xs : List α) :
    cancel c (cancel c xs) = cancel c xs
```

In words: for any "these two cancel" test `c` and any list `xs`, cancelling the
already cancelled list gives the same list. There are no conditions before the
colon, so it holds for every circuit.

## Check it yourself

You need Lean (install `elan`). The commands only work from inside the `lean`
folder, so go there first. From the top of the repository:

```
cd rust/limen-pass/proofs/cancel-adjacent/lean
lake build
lake env lean Audit.lean
```

Expected output:

```
'CancelSpec.cancel_never_grows' depends on axioms: [propext, Quot.sound]
'CancelSpec.cancel_only_removes' depends on axioms: [propext]
'CancelSpec.cancel_idempotent' depends on axioms: [propext, Quot.sound]
'CancelSpec.cancel_preserves_meaning' depends on axioms: [propext]
```

`propext` and `Quot.sound` are two of Lean's standard axioms. If `sorryAx`
ever appears in that list, a proof has a hole. The first build downloads
Mathlib and builds Aeneas: about 10 minutes and several GB.

## How it is kept honest

A nightly job rebuilds the proofs and fails if any headline theorem rests on
an unfinished proof. Each night's result is written to a signed log.

That job runs on the author's machine, so you cannot see it. A public CI check
is planned and not done yet. Until then, run the commands above.

## Versions

Everything is pinned, so a build today and a build next year check the same thing.

| Tool | Version | Pinned in |
|---|---|---|
| Lean | 4.35.0 | `lean/lean-toolchain` |
| Aeneas Lean library | `build-2b9e3ea` | `lean/lakefile.toml` |
| hax Lean library | `v0.3.41` | `lean/lakefile.toml` |
| Mathlib | exact commit | `lean/lake-manifest.json` |

## How it was made

- The Rust was translated to Lean with hax (Charon and Aeneas).
- The model and proofs were written with an AI assistant and checked by Lean.
  A proof checker confirms that a proof is valid. It cannot confirm that the
  claim says what you meant. Reading each claim and checking that was the
  author's job.

## Layout

| Path | What it is |
|---|---|
| `lean/CancelAdjacent/Extraction/` | Generated from the Rust. Overwritten on each extraction. |
| `lean/CancelAdjacent/Verification/Spec.lean` | The model and the four proofs. Hand-written. |
| `lean/CancelAdjacent/Verification/ProofObligations.lean` | Notes on what is still open. |
| `lean/Audit.lean` | Prints what each theorem rests on. |
| `../hax.toml` | How the extraction is run. |

## License

Same as the rest of LIMEN: Elastic License 2.0.
