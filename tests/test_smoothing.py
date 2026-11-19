"""Smoothing (debounce) tests."""

from hand_controller.core import State
from hand_controller.smoothing import Smoother


def feed(sm: Smoother, states):
    return [sm.update(s, 0.9)[0] for s in states]


def test_starts_in_stop():
    assert Smoother().state == State.STOP


def test_direction_needs_four_consecutive_frames_by_default():
    sm = Smoother()
    out = feed(sm, [State.RIGHT] * 4)
    assert out[:3] == [State.STOP] * 3
    assert out[3] == State.RIGHT


def test_stop_needs_two_frames_when_moving():
    sm = Smoother(direction_frames=4, stop_frames=2)
    feed(sm, [State.LEFT] * 4)
    out = feed(sm, [State.STOP, State.STOP])
    assert out == [State.LEFT, State.STOP]


def test_interrupted_streak_restarts_count():
    sm = Smoother(direction_frames=3)
    out = feed(sm, [State.UP, State.UP, State.STOP, State.UP, State.UP])
    assert out == [State.STOP] * 5


def test_changing_candidate_restarts_count():
    sm = Smoother(direction_frames=3)
    out = feed(sm, [State.UP, State.UP, State.DOWN, State.DOWN, State.DOWN])
    assert out[-1] == State.DOWN
    assert State.UP not in out


def test_custom_frame_counts_are_respected():
    sm = Smoother(direction_frames=1, stop_frames=1)
    assert feed(sm, [State.RIGHT]) == [State.RIGHT]
    assert feed(sm, [State.STOP]) == [State.STOP]


def test_confidence_follows_latest_frame_for_current_state():
    sm = Smoother(direction_frames=1)
    sm.update(State.RIGHT, 0.5)
    _, conf = sm.update(State.RIGHT, 0.8)
    assert conf == 0.8
