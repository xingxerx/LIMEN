import pytest

from limen.qfltl.guard import (
    Event,
    bound_awareness_cues,
    check_awareness_never_during_session,
    check_practice_run_cues,
    check_wbtb_lifecycle_within_session,
    sanitize_awareness,
    sanitize_wbtb,
)


def test_sanitize_awareness_clamps_into_bounds():
    d, i, s = sanitize_awareness(duration_seconds=-5, interval_seconds=0, cue_at_start=True)
    assert d == 300 and 5 <= i <= 60 and s is True
    d2, i2, _ = sanitize_awareness(duration_seconds=60, interval_seconds=1800, cue_at_start=False)
    assert d2 == 60 and i2 == 30  # min(60, 60/2)


def test_sanitize_wbtb_clamps():
    e, m, h, mo = sanitize_wbtb(True, "bad", hours_after_onset=20.0, minute_of_day=5000)
    assert e is True and m == "after_sleep_onset" and 1.0 <= h <= 10.0 and 0 <= mo < 1440
    e2, m2, h2, mo2 = sanitize_wbtb(False, "clock_time", 2.5, 6 * 60 + 30)
    assert (e2, m2, h2, mo2) == (False, "clock_time", 2.5, 390)


def test_awareness_never_during_session_positive():
    # Practice cues before and after a session window are allowed; none inside.
    ev = [
        Event(ts=0.0, kind="qfltl.pulse.cue", mode="practice"),
        Event(ts=10.0, kind="session_start"),
        Event(ts=20.0, kind="qfltl.pulse.cue", mode="night"),
        Event(ts=30.0, kind="session_end"),
        Event(ts=40.0, kind="qfltl.pulse.cue", mode="practice"),
    ]
    check_awareness_never_during_session(ev)


def test_awareness_never_during_session_negative():
    ev = [
        Event(ts=0.0, kind="session_start"),
        Event(ts=1.0, kind="qfltl.pulse.cue", mode="practice"),
        Event(ts=2.0, kind="session_end"),
    ]
    with pytest.raises(AssertionError):
        check_awareness_never_during_session(ev)


def test_wbtb_lifecycle_within_session_and_not_ending():
    ev = [
        Event(ts=0.0, kind="session_start"),
        Event(ts=1.0, kind="wbtb_wake"),
        Event(ts=2.0, kind="wbtb_start"),
        Event(ts=6.0, kind="wbtb_resume"),
        Event(ts=10.0, kind="session_end"),
    ]
    check_wbtb_lifecycle_within_session(ev)


def test_wbtb_negative_outside_session():
    ev = [Event(ts=0.0, kind="wbtb_wake")]
    with pytest.raises(AssertionError):
        check_wbtb_lifecycle_within_session(ev)


def test_cues_bounded():
    # 5-minute timer, 60s interval, start cue: 1(start) + floor(299/60) = 1 + 4 = 5
    bound = bound_awareness_cues(duration_seconds=300, interval_seconds=60, cue_at_start=True)
    assert bound == 5
    # Simulated run with exactly bound cues passes
    start, end = 0.0, 300.0
    events = [Event(ts=t, kind="qfltl.pulse.cue", mode="practice") for t in [0, 60, 120, 180, 240]]
    check_practice_run_cues(events, start_ts=start, end_ts=end, duration_seconds=300, interval_seconds=60, cue_at_start=True)
    # Negative: one extra cue trips the guard
    events2 = events + [Event(ts=245.0, kind="qfltl.pulse.cue", mode="practice")]
    with pytest.raises(AssertionError):
        check_practice_run_cues(events2, start_ts=start, end_ts=end, duration_seconds=300, interval_seconds=60, cue_at_start=True)
