from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


# QFLTL guard for Oneiro: temporal and settings checks used by offline tests.
#
# Events follow the Oneiro watch log kinds:
# - "session_start", "session_end"
# - "qfltl.pulse.cue" / "qfltl.pulse.confirm" with mode "night" or "practice"
# - "wbtb_wake", "wbtb_start", "wbtb_resume"
#
# We model only the invariants we need to check; no network or OS calls.


@dataclass(frozen=True)
class Event:
    ts: float  # seconds from epoch for ordering (monotone increasing in a single test)
    kind: str
    mode: Optional[str] = None  # "night" or "practice" for qfltl.pulse
    session_id: Optional[str] = None


def _session_intervals(events: Iterable[Event]) -> List[Tuple[float, Optional[float], Optional[str]]]:
    """Return [(start_ts, end_ts, sid)] intervals for each session."""
    out: List[Tuple[float, Optional[float], Optional[str]]] = []
    current: Optional[Tuple[float, Optional[float], Optional[str]]] = None
    for e in sorted(events, key=lambda x: x.ts):
        if e.kind == "session_start":
            # Close any open session without an end (defensive)
            if current is not None and current[1] is None:
                out.append(current)
            current = (e.ts, None, e.session_id)
        elif e.kind == "session_end":
            if current is not None and current[1] is None:
                current = (current[0], e.ts, current[2])
                out.append(current)
                current = None
    if current is not None:
        out.append(current)
    return out


def _in_session_at(ts: float, intervals: List[Tuple[float, Optional[float], Optional[str]]]) -> bool:
    for (s, e, _) in intervals:
        if ts >= s and (e is None or ts <= e):
            return True
    return False


def check_awareness_never_during_session(events: Iterable[Event]) -> None:
    """Assert that practice cues never occur during an active sleep session."""
    intervals = _session_intervals(events)
    for e in events:
        if e.kind.startswith("qfltl.pulse") and e.mode == "practice":
            assert not _in_session_at(e.ts, intervals), "practice cue during active session"


def sanitize_awareness(duration_seconds: float, interval_seconds: float, cue_at_start: bool) -> Tuple[int, int, bool]:
    """Mirror AwarenessTimerConfig.sanitized() discrete clamps."""
    d = int(duration_seconds) if duration_seconds == duration_seconds and duration_seconds > 0 else 300
    if d < 10:
        d = 300
    i = int(interval_seconds) if interval_seconds == interval_seconds and interval_seconds > 0 else 60
    if i >= d:
        i = max(5, min(60, d // 2))
    return d, i, bool(cue_at_start)


def sanitize_wbtb(enabled: bool, mode: str, hours_after_onset: float, minute_of_day: int) -> Tuple[bool, str, float, int]:
    """Mirror QfltlWbtbConfig.sanitized() clamps."""
    m = mode if mode in ("after_sleep_onset", "clock_time") else "after_sleep_onset"
    h = hours_after_onset if (hours_after_onset == hours_after_onset and 1.0 <= hours_after_onset <= 10.0) else 4.5
    mo = minute_of_day if 0 <= int(minute_of_day) < 24 * 60 else 5 * 60
    return bool(enabled), m, float(h), int(mo)


def check_wbtb_lifecycle_within_session(events: Iterable[Event]) -> None:
    """Assert that each wbtb_* lifecycle event occurs during an active session and does not end it."""
    intervals = _session_intervals(events)
    for e in events:
        if e.kind in ("wbtb_wake", "wbtb_start", "wbtb_resume"):
            assert _in_session_at(e.ts, intervals), f"{e.kind} outside session"
    # WBTB sequence must not end the session itself: ensure there is no immediate session_end at same timestamp
    by_ts = {}
    for e in events:
        by_ts.setdefault(e.ts, []).append(e.kind)
    for ts, kinds in by_ts.items():
        if any(k in ("wbtb_wake", "wbtb_start") for k in kinds):
            assert "session_end" not in kinds, "WBTB must not end the session"


def bound_awareness_cues(duration_seconds: int, interval_seconds: int, cue_at_start: bool) -> int:
    """Return an upper bound on the number of cues: floor((duration-1)/interval) + start?1:0."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if duration_seconds <= 0:
        return 0
    positive = (max(0, duration_seconds - 1)) // interval_seconds
    return (1 if cue_at_start else 0) + positive


def check_practice_run_cues(events: Iterable[Event], start_ts: float, end_ts: float, duration_seconds: int,
                            interval_seconds: int, cue_at_start: bool) -> None:
    """Check that a practice run's cues do not exceed the theoretical bound."""
    bound = bound_awareness_cues(duration_seconds, interval_seconds, cue_at_start)
    cues = [e for e in events if e.mode == "practice" and e.kind == "qfltl.pulse.cue" and start_ts <= e.ts < end_ts]
    assert len(cues) <= bound, f"too many cues: {len(cues)} > {bound}"
