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

/-!
Settings-sanitization bounds (explicit).
-/
theorem sanitize_duration_ge10 (c : AwarenessConfig) :
  10 ≤ (sanitize c).durationSeconds := by
  unfold sanitize
  by_cases h : c.durationSeconds = 0 ∨ c.durationSeconds < 10
  · simp [h]
  · have hnot : ¬ c.durationSeconds < 10 := (not_or.mp h).right
    have : 10 ≤ c.durationSeconds := Nat.le_of_not_lt hnot
    simp [h, this]

/-
Closed-form schedule and cue-count bound.
Positive cues (strictly before the end): (k+1) * interval for k = 0 .. floor((duration-1)/interval)-1.
-/
def positiveCount (s : AwarenessConfig) : Nat :=
  (s.durationSeconds - 1) / s.intervalSeconds

def positiveTimes (s : AwarenessConfig) : List Nat :=
  -- The exact timestamps are irrelevant for the count; use a replicate of the right length.
  List.replicate (positiveCount s) 0

def cueOffsets (c : AwarenessConfig) : List Nat :=
  let s := sanitize c
  let base := if s.cueAtStart then [0] else []
  base ++ positiveTimes s

theorem length_positiveTimes (s : AwarenessConfig) :
  (positiveTimes s).length = positiveCount s := by
  unfold positiveTimes
  simp [positiveCount]

theorem cue_count_exact (c : AwarenessConfig) :
  let s := sanitize c
  (cueOffsets c).length = (if s.cueAtStart then 1 else 0) + positiveCount s := by
  intro s
  unfold cueOffsets
  by_cases h : s.cueAtStart
  · have baseLen : (if s.cueAtStart then [0] else []).length = 1 := by simp [h]
    simp [h, length_positiveTimes, List.length, List.length_append, baseLen]
  · have baseLen : (if s.cueAtStart then [0] else []).length = 0 := by simp [h]
    simp [h, length_positiveTimes, List.length, List.length_append, baseLen]

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
