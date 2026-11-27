/- Audit.lean: the scanner. Prints everything each headline theorem rests on.

Run from this folder:   lake env lean Audit.lean

A clean result lists only `propext`, `Classical.choice` and `Quot.sound`, or
says a theorem depends on no axioms. `sorryAx` means a proof has a hole.

The first four are about the model (Spec.lean). The fifth links the model to
the Rust code: the Aeneas-extracted `limen_pass.cancel_adjacent` returns exactly
`CancelSpec.cancel` (Refinement.lean), so the first four hold for the shipped Rust. -/
import CancelAdjacent

#print axioms CancelSpec.cancel_never_grows
#print axioms CancelSpec.cancel_only_removes
#print axioms CancelSpec.cancel_idempotent
#print axioms CancelSpec.cancel_preserves_meaning
#print axioms CancelRefine.cancel_adjacent_refines
