"""Feature function tests: invariance to position and size, and the expected length."""

import numpy as np

from hand_controller.features import compute_features, feature_names
from tests.synthetic import make_hand, pointing


def test_length_without_and_with_z():
    hand = pointing(0)
    assert compute_features(hand.landmarks, hand.width, hand.height).shape == (42,)
    assert compute_features(hand.landmarks, hand.width, hand.height, use_z=True).shape == (63,)
    assert len(feature_names(False)) == 42 and len(feature_names(True)) == 63


def test_wrist_is_origin_and_palm_scale_is_one():
    hand = pointing(0)
    feats = compute_features(hand.landmarks, hand.width, hand.height).reshape(21, 2)
    assert np.allclose(feats[0], 0.0)
    # Palm size is the wrist-to-middle-MCP distance, so landmark 9 is at length 1 from the wrist.
    assert np.isclose(np.linalg.norm(feats[9]), 1.0)


def test_invariant_to_translation():
    hand = pointing(30)
    shifted = hand.landmarks.copy()
    shifted[:, 0] += 0.05
    shifted[:, 1] -= 0.03
    a = compute_features(hand.landmarks, hand.width, hand.height)
    b = compute_features(shifted, hand.width, hand.height)
    assert np.allclose(a, b, atol=1e-6)


def test_invariant_to_scale():
    hand = make_hand(60, [True, False, False, False])
    # Scaling about the wrist changes the hand's size but not its shape.
    scaled = hand.landmarks.copy()
    wrist = scaled[0, :2].copy()
    scaled[:, :2] = wrist + (scaled[:, :2] - wrist) * 1.5
    a = compute_features(hand.landmarks, hand.width, hand.height)
    b = compute_features(scaled, hand.width, hand.height)
    assert np.allclose(a, b, atol=1e-6)


def test_degenerate_hand_gives_finite_values():
    landmarks = np.zeros((21, 3))
    assert np.all(np.isfinite(compute_features(landmarks, 640, 480)))
