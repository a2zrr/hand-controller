"""Gesture classifiers. Every classifier takes a Hand (or None) and returns (State, confidence)."""

from __future__ import annotations

import math
from typing import Protocol

import joblib
import numpy as np

from hand_controller.config import ClassifierConfig, FeaturesConfig, resolve_project_path
from hand_controller.core import Hand, State
from hand_controller.features import hand_features

# Landmark indexes (MediaPipe hand model).
WRIST = 0
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
# (pip, tip) for index, middle, ring, pinky. The thumb is ignored on purpose.
FINGERS: dict[str, tuple[int, int]] = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}
# Direction of each state in degrees, math convention (counter-clockwise, y up).
DIRECTION_ANGLES: dict[State, float] = {
    State.RIGHT: 0.0,
    State.UP: 90.0,
    State.LEFT: 180.0,
    State.DOWN: -90.0,
}


class Classifier(Protocol):
    """Interface every gesture classifier implements."""

    def classify(self, hand: Hand | None) -> tuple[State, float]:
        """Return the raw (unsmoothed) state and a confidence 0..1 for one frame."""
        ...


def _wrap_degrees(angle: float) -> float:
    """Map any angle to the range (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def finger_extended(pixels: np.ndarray, pip: int, tip: int, ratio: float) -> bool:
    """A finger is extended if its tip is clearly farther from the wrist than its PIP joint.

    Distances to the wrist do not depend on the hand's rotation, so this still works when
    the hand points sideways or upside down.
    """
    wrist = pixels[WRIST]
    tip_dist = float(np.linalg.norm(pixels[tip] - wrist))
    pip_dist = float(np.linalg.norm(pixels[pip] - wrist))
    return tip_dist > pip_dist * ratio


def pointing_angle(pixels: np.ndarray) -> float:
    """Angle in degrees of the vector from index MCP to index tip (y up, 0 = right)."""
    dx = float(pixels[INDEX_TIP, 0] - pixels[INDEX_MCP, 0])
    dy = float(pixels[INDEX_TIP, 1] - pixels[INDEX_MCP, 1])
    # Image y points down, so negate dy to get a standard math angle.
    return math.degrees(math.atan2(-dy, dx))


class RuleClassifier:
    """Hand-written rules: pointing index finger gives a direction, anything else gives STOP."""

    def __init__(self, cfg: ClassifierConfig) -> None:
        if not 0.0 <= cfg.dead_zone_deg < 90.0:
            raise ValueError("dead_zone_deg must be in [0, 90)")
        self.cfg = cfg
        self.dead_zone_state = State(cfg.dead_zone_state)

    def classify(self, hand: Hand | None) -> tuple[State, float]:
        """Classify one hand. No hand or a low-score hand is STOP with confidence 0."""
        if hand is None or hand.score < self.cfg.min_hand_score:
            return State.STOP, 0.0

        pixels = hand.pixels()
        ratio = self.cfg.finger_extend_ratio
        index_up = finger_extended(pixels, *FINGERS["index"], ratio)
        others_up = [finger_extended(pixels, *FINGERS[name], ratio) for name in ("middle", "ring", "pinky")]

        if index_up and not any(others_up):
            return self._direction(pixels, hand.score)
        if index_up and all(others_up):
            # All four fingers extended: open palm.
            return State.STOP, hand.score
        if not index_up and not any(others_up):
            # Fist: clearly STOP.
            return State.STOP, hand.score
        # Mixed finger pattern (e.g. peace sign): ambiguous, so STOP with lower confidence.
        return State.STOP, 0.5 * hand.score

    def _direction(self, pixels: np.ndarray, score: float) -> tuple[State, float]:
        """Pick RIGHT/LEFT/UP/DOWN from the pointing angle, or the dead-zone state."""
        angle = pointing_angle(pixels)
        best = min(DIRECTION_ANGLES, key=lambda s: abs(_wrap_degrees(angle - DIRECTION_ANGLES[s])))
        diff = abs(_wrap_degrees(angle - DIRECTION_ANGLES[best]))
        half_sector = 45.0 - self.cfg.dead_zone_deg / 2.0
        if diff > half_sector:
            return self.dead_zone_state, 0.5 * score
        alignment = 1.0 - diff / half_sector  # 1 = pointing exactly at the sector centre
        return best, score * (0.5 + 0.5 * alignment)


class ModelClassifier:
    """Trained scikit-learn model (see tools/train.py). Same interface as RuleClassifier."""

    def __init__(self, cfg: ClassifierConfig, features: FeaturesConfig) -> None:
        path = resolve_project_path(cfg.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. Train one with `uv run python tools/train.py`, "
                "or run with --classifier rules."
            )
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.use_z = bool(bundle["use_z"])
        if self.use_z != features.use_z:
            raise ValueError(
                f"The model was trained with use_z={self.use_z} but [features] use_z is "
                f"{features.use_z}. Change the config or retrain the model."
            )
        self.min_confidence = cfg.min_model_confidence
        self.min_hand_score = cfg.min_hand_score

    def classify(self, hand: Hand | None) -> tuple[State, float]:
        """Predict the state from the feature vector. Low confidence becomes STOP."""
        if hand is None or hand.score < self.min_hand_score:
            return State.STOP, 0.0
        x = hand_features(hand, self.use_z)[None, :]
        proba = self.pipeline.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx]) * hand.score
        state = State(self.pipeline.classes_[idx])
        if proba[idx] < self.min_confidence:
            return State.STOP, confidence
        return state, confidence


def build_classifier(cfg: ClassifierConfig, features: FeaturesConfig) -> Classifier:
    """Create the classifier named in the config (`rules` or `model`)."""
    if cfg.name == "rules":
        return RuleClassifier(cfg)
    if cfg.name == "model":
        return ModelClassifier(cfg, features)
    raise ValueError(f"Unknown classifier '{cfg.name}'. Use 'rules' or 'model'.")
