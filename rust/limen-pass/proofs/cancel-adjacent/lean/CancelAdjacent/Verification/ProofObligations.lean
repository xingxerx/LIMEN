/- Handwritten proofs about the extracted definitions.

hax creates this file once and never modifies anything under
`Verification/`. -/
import CancelAdjacent.Extraction
import CancelAdjacent.Verification.Spec

/-!
# What is proven, and what is not yet

Proven in Spec.lean, for the model `CancelSpec.cancel`:

0. `cancel_never_grows`: the output is never longer than the input.
1. `cancel_only_removes`: the output is the input with items taken out.
   Nothing is added and nothing changes order.
2. `cancel_idempotent`: running the pass twice equals running it once.
3. `cancel_preserves_meaning`: the meaning is unchanged, given that each pair
   the pass may cancel does nothing when run back to back.

Not proven yet: that the Rust function `limen_pass.cancel_adjacent` (extracted
in Extraction/Funs.lean) computes exactly `CancelSpec.cancel`.

Until that link is proven, the four claims are about the model. The Rust code
is tied to the model by reading the two side by side, and by the property test
`cancel_adjacent_matches_the_old_loop` in the Rust crate.

The statement still to prove, for a `clone` that returns its argument and a
`cancels` that always answers with a plain function `c`:

    limen_pass.cancel_adjacent cloneInst cancelsInst ops
      succeeds with a vector whose contents are `CancelSpec.cancel c ops.val`

Do not add that theorem with `sorry`. A theorem with a hole would make the
nightly check report something that is not true.
-/
