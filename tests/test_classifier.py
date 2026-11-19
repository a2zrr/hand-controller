"""Rule classifier tests on synthetic landmarks."""

import pytest

from hand_controller.classifier import RuleClassifier
from hand_controller.config import ClassifierConfig
from hand_controller.core import State
from tests.synthetic import fist, make_hand, open_palm, pointing


@pytest.fixture
def clf() -> RuleClassifier:
    return RuleClassifier(ClassifierConfig())


@pytest.mark.parametrize(
    "angle, expected",
    [(0, State.RIGHT), (90, State.UP), (180, State.LEFT), (-90, State.DOWN)],
)
def test_pointing_in_four_directions(clf, angle, expected):
    state, conf = clf.classify(pointing(angle))
    assert state == expected
    assert 0.0 < conf <= 1.0


@pytest.mark.parametrize("angle, expected", [(10, State.RIGHT), (80, State.UP), (-170, State.LEFT), (-100, State.DOWN)])
def test_pointing_near_centre_of_sector(clf, angle, expected):
    assert clf.classify(pointing(angle))[0] == expected


@pytest.mark.parametrize("angle", [45, 135, -135, -45])
def test_diagonal_falls_in_dead_zone_and_gives_stop(clf, angle):
    assert clf.classify(pointing(angle))[0] == State.STOP


def test_dead_zone_state_is_configurable():
    clf = RuleClassifier(ClassifierConfig(dead_zone_state="STOP"))
    assert clf.dead_zone_state == State.STOP


def test_open_palm_is_stop(clf):
    state, conf = clf.classify(open_palm())
    assert state == State.STOP
    assert conf > 0.5


def test_fist_is_stop(clf):
    state, conf = clf.classify(fist())
    assert state == State.STOP
    assert conf > 0.5


def test_mixed_fingers_are_ambiguous_stop(clf):
    # Index and middle extended (peace sign): ambiguous, so STOP with lower confidence.
    hand = make_hand(90.0, [True, True, False, False])
    state, conf = clf.classify(hand)
    assert state == State.STOP
    assert conf < 0.5


@pytest.mark.parametrize("angle", [0, 90, 180, -90])
def test_pointing_rotated_still_works(clf, angle):
    # A hand pointing sideways is rotated by 90 degrees: the finger test must not depend on orientation.
    assert clf.classify(pointing(angle))[0] != State.STOP


def test_no_hand_is_stop_with_zero_confidence(clf):
    assert clf.classify(None) == (State.STOP, 0.0)


def test_low_score_hand_is_treated_as_no_hand(clf):
    hand = pointing(0)
    weak = type(hand)(hand.landmarks, hand.handedness, 0.2, hand.width, hand.height)
    assert clf.classify(weak) == (State.STOP, 0.0)


def test_bad_dead_zone_is_rejected():
    with pytest.raises(ValueError):
        RuleClassifier(ClassifierConfig(dead_zone_deg=95.0))
