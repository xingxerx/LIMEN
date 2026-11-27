/-
CancelAdjacent.Verification.Spec

The algorithm behind `cancel_adjacent` (rust/limen-pass/src/lib.rs), written as a
small Lean model, with four proven claims about it.

This file is hand-written and uses core Lean only: no math library, no Aeneas.
It works for any kind of item and any "these two cancel" test `c`. In LIMEN the
items are gates and `c` is `self_inverse_pair`.

What is proven here is about the model below. The link from the Rust code to
this model is a separate statement (see ProofObligations.lean).
-/

namespace CancelSpec

variable {α : Type}

/-- One step. Look at the item on top of the stack. If it cancels with `x`, drop
the top. Otherwise push `x`. The head of the list is the newest item. -/
def step (c : α → α → Bool) (st : List α) (x : α) : List α :=
  match st with
  | [] => [x]
  | top :: rest => if c top x then rest else x :: top :: rest

/-- Walk the input once, starting from an empty stack. Newest item first. -/
def cancelStack (c : α → α → Bool) (xs : List α) : List α :=
  xs.foldl (step c) []

/-- The pass: the stack put back in circuit order. -/
def cancel (c : α → α → Bool) (xs : List α) : List α :=
  (cancelStack c xs).reverse

/-! ## Claim 1: the pass never makes a circuit longer -/

theorem foldl_step_length_le (c : α → α → Bool) (xs st : List α) :
    (xs.foldl (step c) st).length ≤ st.length + xs.length := by
  induction xs generalizing st with
  | nil => simp
  | cons x xs ih =>
    have h1 : (step c st x).length ≤ st.length + 1 := by
      cases st with
      | nil => simp [step]
      | cons top rest =>
        cases hcx : c top x with
        | true => simp [step, hcx]; omega
        | false => simp [step, hcx]
    have h2 := ih (step c st x)
    simp only [List.foldl_cons, List.length_cons]
    omega

/-- Claim 1. The output is never longer than the input. -/
theorem cancel_never_grows (c : α → α → Bool) (xs : List α) :
    (cancel c xs).length ≤ xs.length := by
  have h := foldl_step_length_le c xs []
  simpa [cancel, cancelStack] using h

/-! ## Claim 2: the pass only removes items -/

theorem foldl_step_sublist (c : α → α → Bool) (xs st : List α) :
    List.Sublist (xs.foldl (step c) st).reverse (st.reverse ++ xs) := by
  induction xs generalizing st with
  | nil => simp
  | cons x xs ih =>
    have h1 : List.Sublist ((step c st x).reverse ++ xs) (st.reverse ++ x :: xs) := by
      cases st with
      | nil => simp [step]
      | cons top rest =>
        cases hcx : c top x with
        | true => simp [step, hcx, List.append_assoc]
        | false => simp [step, hcx]
    simpa using List.Sublist.trans (ih (step c st x)) h1

/-- Claim 2. The output is the input with some items taken out. Nothing is added
and nothing changes order. -/
theorem cancel_only_removes (c : α → α → Bool) (xs : List α) :
    List.Sublist (cancel c xs) xs := by
  have h := foldl_step_sublist c xs []
  simpa [cancel, cancelStack] using h

/-! ## Claim 3: running the pass twice changes nothing more -/

/-- No two neighbours on the stack cancel. The head is the newer item. -/
def Settled (c : α → α → Bool) : List α → Prop
  | [] => True
  | [_] => True
  | newer :: older :: rest => c older newer = false ∧ Settled c (older :: rest)

theorem settled_tail (c : α → α → Bool) {x : α} {st : List α}
    (h : Settled c (x :: st)) : Settled c st := by
  cases st with
  | nil => simp [Settled]
  | cons y ys =>
    simp [Settled] at h
    exact h.2

theorem settled_step (c : α → α → Bool) (st : List α) (x : α)
    (h : Settled c st) : Settled c (step c st x) := by
  cases st with
  | nil => simp [step, Settled]
  | cons top rest =>
    cases hcx : c top x with
    | true => simpa [step, hcx] using settled_tail c h
    | false => simpa [step, hcx, Settled] using h

theorem settled_foldl (c : α → α → Bool) (xs st : List α)
    (h : Settled c st) : Settled c (xs.foldl (step c) st) := by
  induction xs generalizing st with
  | nil => simpa using h
  | cons x xs ih => simpa using ih (step c st x) (settled_step c st x h)

/-- A settled stack is left alone by the pass. -/
theorem cancelStack_reverse_of_settled (c : α → α → Bool) (st : List α)
    (h : Settled c st) : cancelStack c st.reverse = st := by
  induction st with
  | nil => simp [cancelStack]
  | cons newer older ih =>
    have hrest : Settled c older := settled_tail c h
    have e : cancelStack c (newer :: older).reverse
        = step c (cancelStack c older.reverse) newer := by
      simp [cancelStack, List.foldl_append]
    rw [e, ih hrest]
    cases older with
    | nil => simp [step]
    | cons o os =>
      have hc : c o newer = false := by
        simp [Settled] at h
        exact h.1
      simp [step, hc]

/-- Claim 3. Running the pass on its own output gives the same output. -/
theorem cancel_idempotent (c : α → α → Bool) (xs : List α) :
    cancel c (cancel c xs) = cancel c xs := by
  have hs : Settled c (cancelStack c xs) := by
    unfold cancelStack
    exact settled_foldl c xs [] (by simp [Settled])
  unfold cancel
  rw [cancelStack_reverse_of_settled c (cancelStack c xs) hs]

/-! ## Claim 4: the pass does not change what the circuit means -/

/-- What a circuit means, kept abstract. Each item has a meaning, meanings
combine in order, and there is a "do nothing" meaning. -/
structure Meaning (α : Type) (M : Type) where
  den : α → M
  mul : M → M → M
  one : M
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a

variable {M : Type}

/-- The meaning of a whole circuit: combine the meanings of its items in order. -/
def Meaning.ofList (m : Meaning α M) : List α → M
  | [] => m.one
  | x :: xs => m.mul (m.den x) (m.ofList xs)

theorem Meaning.ofList_append (m : Meaning α M) (l r : List α) :
    m.ofList (l ++ r) = m.mul (m.ofList l) (m.ofList r) := by
  induction l with
  | nil => simp [Meaning.ofList, m.one_mul]
  | cons x l ih => simp [Meaning.ofList, ih, m.mul_assoc]

theorem foldl_step_meaning (m : Meaning α M) (c : α → α → Bool)
    (hc : ∀ a b, c a b = true → m.mul (m.den a) (m.den b) = m.one)
    (xs st : List α) :
    m.ofList (xs.foldl (step c) st).reverse
      = m.mul (m.ofList st.reverse) (m.ofList xs) := by
  induction xs generalizing st with
  | nil => simp [Meaning.ofList, m.mul_one]
  | cons x xs ih =>
    rw [List.foldl_cons, ih (step c st x)]
    cases st with
    | nil => simp [step, Meaning.ofList, m.one_mul, m.mul_one]
    | cons top rest =>
      cases hcx : c top x with
      | true =>
        have hinv : ∀ z, m.mul (m.den top) (m.mul (m.den x) z) = z := by
          intro z
          rw [← m.mul_assoc, hc top x hcx, m.one_mul]
        simp [step, hcx, Meaning.ofList, Meaning.ofList_append, m.mul_assoc,
          m.mul_one, hinv]
      | false =>
        simp [step, hcx, Meaning.ofList, Meaning.ofList_append, m.mul_assoc,
          m.mul_one]

/-- Claim 4. If every pair the pass may cancel really does nothing when run back
to back, then the pass never changes what the circuit means. -/
theorem cancel_preserves_meaning (m : Meaning α M) (c : α → α → Bool)
    (hc : ∀ a b, c a b = true → m.mul (m.den a) (m.den b) = m.one)
    (xs : List α) :
    m.ofList (cancel c xs) = m.ofList xs := by
  have h := foldl_step_meaning m c hc xs []
  simpa [cancel, cancelStack, Meaning.ofList, m.one_mul] using h

end CancelSpec
