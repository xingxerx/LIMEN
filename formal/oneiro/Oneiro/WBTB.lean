namespace Oneiro

/-
QFLTL WBTB: configuration sanitization and a minimal event-guard.

Swift reference: oneiro_watch/Source/Logic/QfltlWbtbLogic.swift
  struct QfltlWbtbConfig { enabled: Bool, mode: "after_sleep_onset"|"clock_time",
                            hoursAfterOnset: Double (default 4.5), minuteOfDay: Int (0..1439) }
  func sanitized(): clamp hours to [1,10], minuteOfDay to [0,1439], mode to a known value.
-/

-- No sorry used in this module.

inductive WbtbMode | afterSleepOnset | clockTime
deriving DecidableEq, Repr

structure WbtbConfig where
  enabled        : Bool
  mode           : WbtbMode
  hoursAfterOnset : Nat   -- tenth-hours omitted in this Nat model; safe discrete bound proofs only
  minuteOfDay     : Nat   -- 0..1439
deriving DecidableEq, Repr

namespace WbtbConfig

def sanitize (c : WbtbConfig) : WbtbConfig :=
  let hours := if c.hoursAfterOnset < 1 then 4 else if c.hoursAfterOnset > 10 then 4 else c.hoursAfterOnset
  let minute := if c.minuteOfDay ≥ 24*60 then 5*60 else c.minuteOfDay
  { enabled := c.enabled, mode := c.mode, hoursAfterOnset := hours, minuteOfDay := minute }

end WbtbConfig

/-
Minimal event guard: WBTB lifecycle events only occur during an active session.
-/
inductive Ev
| session_start
| session_end
| wbtb_wake
| wbtb_start
| wbtb_resume
deriving DecidableEq, Repr

structure Sess where
  inSession : Bool := false
deriving DecidableEq, Repr

def step (s : Sess) (e : Ev) : Option Sess :=
  match e with
  | .session_start => some { inSession := true }
  | .session_end   => if s.inSession then some { inSession := false } else none
  | .wbtb_wake | .wbtb_start | .wbtb_resume =>
      if s.inSession then some s else none

theorem wbtb_requires_session (s : Sess) :
  step s Ev.wbtb_wake = none ↔ s.inSession = false := by
  by_cases h : s.inSession <;> simp [step, h]

-- Additional lifecycle facts are straightforward from `step` by case analysis and `simp`.

end Oneiro
