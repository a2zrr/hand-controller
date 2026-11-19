"""Time-based smoothing tests. Frames are fed with explicit timestamps (ms)."""

from hand_controller.core import State
from hand_controller.smoothing import Smoother


def feed(sm: Smoother, states, frame_ms: float = 40.0, start_ms: float = 0.0):
    """Feed one state per frame, `frame_ms` apart, and return the smoothed outputs."""
    out = []
    for i, s in enumerate(states):
        out.append(sm.update(s, 0.9, start_ms + i * frame_ms)[0])
    return out


def test_starts_in_stop():
    assert Smoother().state == State.STOP


def test_direction_needs_120ms_by_default_at_25fps():
    # At 25 FPS (40 ms per frame) a direction is first seen at 0 ms, accepted at 120 ms (frame 4).
    out = feed(Smoother(), [State.RIGHT] * 5)
    assert out[:3] == [State.STOP] * 3
    assert out[3] == State.RIGHT


def test_stop_needs_60ms_by_default():
    sm = Smoother(direction_ms=120, stop_ms=60)
    feed(sm, [State.LEFT] * 5)
    # STOP first seen at 200 ms. At 240 ms only 40 ms have passed (not enough);
    # at 280 ms, 80 ms have passed (>= 60), so STOP is accepted on the third frame.
    out = feed(sm, [State.STOP] * 3, start_ms=200)
    assert out == [State.LEFT, State.LEFT, State.STOP]


def test_same_real_time_gives_same_result_at_different_frame_rates():
    # 120 ms of stable RIGHT must be accepted whether frames come every 40 ms or every 20 ms.
    for frame_ms in (40.0, 20.0, 60.0):
        sm = Smoother(direction_ms=120)
        t = 0.0
        accepted_at = None
        while t <= 300:
            if sm.update(State.RIGHT, 0.9, t)[0] == State.RIGHT and accepted_at is None:
                accepted_at = t
            t += frame_ms
        assert accepted_at is not None
        assert 120 <= accepted_at < 120 + frame_ms


def test_interrupted_streak_restarts_the_timer():
    sm = Smoother(direction_ms=120)
    out = feed(sm, [State.UP, State.UP, State.STOP, State.UP, State.UP], frame_ms=40)
    assert State.UP not in out


def test_changing_candidate_restarts_the_timer():
    sm = Smoother(direction_ms=120)
    out = feed(sm, [State.UP, State.UP, State.DOWN, State.DOWN, State.DOWN, State.DOWN], frame_ms=40)
    assert out[-1] == State.DOWN
    assert State.UP not in out


def test_zero_times_accept_immediately():
    sm = Smoother(direction_ms=0, stop_ms=0)
    assert feed(sm, [State.RIGHT]) == [State.RIGHT]
    assert feed(sm, [State.STOP], start_ms=100) == [State.STOP]


def test_negative_times_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        Smoother(direction_ms=-1)


def test_confidence_follows_latest_frame_for_current_state():
    sm = Smoother(direction_ms=0)
    sm.update(State.RIGHT, 0.5, 0)
    _, conf = sm.update(State.RIGHT, 0.8, 40)
    assert conf == 0.8
