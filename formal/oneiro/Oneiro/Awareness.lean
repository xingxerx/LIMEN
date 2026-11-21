namespace Oneiro

/-
Awareness Timer model and safety properties, based on oneiro_watch/Source/Logic/AwarenessTimer.swift.

Swift reference:
  struct AwarenessTimerConfig { durationSeconds: Double, intervalSeconds: Double, cueAtStart: Bool }
  func sanitized():
    if durationSeconds not finite or < 10 -> durationSeconds := 300
    if intervalSeconds not finite or ≤ 0 -> intervalSeconds := 60
    if intervalSeconds ≥ durationSeconds -> intervalSeconds := max(5, min(60, durationSeconds / 2))
  enum AwarenessTimerSchedule.cueOffsets(config): returns offsets strictly less than duration; includes 0 when cueAtStart.
-/

-- No sorry used in this module.

structure AwarenessConfig where
  durationSeconds : Nat
  intervalSeconds : Nat
  cueAtStart      : Bool
deriving DecidableEq, Repr

namespace AwarenessConfig

def sanitize (c : AwarenessConfig) : AwarenessConfig :=
  let duration :=
    if c.durationSeconds = 0 ∨ c.durationSeconds < 10 then 300 else c.durationSeconds
  let interval0 := if c.intervalSeconds = 0 then 60 else c.intervalSeconds
  let interval :=
    if interval0 ≥ duration then
      Nat.max 5 (Nat.min 60 (duration / 2))
    else interval0
  { durationSeconds := duration, intervalSeconds := interval, cueAtStart := c.cueAtStart }

/- Session-activity guard: the timer never starts while a session is active. -/
structure TimerState where
  running      : Bool := false
  scheduledCues : Nat := 0
deriving Repr, DecidableEq

def startTimer (sessionActive : Bool) (cfg : AwarenessConfig) (st : TimerState := {}) : TimerState :=
  if sessionActive then st
  else
    { running := true, scheduledCues := 0 }

theorem start_blocked_when_session_active (cfg : AwarenessConfig) (st : TimerState) :
  startTimer true cfg st = st := by
  simp [startTimer]

theorem start_schedules_when_inactive (cfg : AwarenessConfig) :
  let s := startTimer false cfg {}
  s.running = true := by
  simp [startTimer]

end AwarenessConfig

end Oneiro
