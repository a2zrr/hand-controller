"""Shared data types used by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class State(str, Enum):
    """The five states a game can receive."""

    RIGHT = "RIGHT"
    LEFT = "LEFT"
    UP = "UP"
    DOWN = "DOWN"
    STOP = "STOP"


@dataclass(frozen=True)
class Hand:
    """One detected hand, in the coordinates of the mirrored camera image.

    Attributes:
        landmarks: Array of shape (21, 3) with normalized x, y, z (x and y in 0..1).
        handedness: "Left" or "Right", the label of the person's real hand.
        score: Detector confidence for the handedness label, 0..1.
        width: Frame width in pixels the landmarks were produced from.
        height: Frame height in pixels the landmarks were produced from.
    """

    landmarks: np.ndarray
    handedness: str
    score: float
    width: int
    height: int

    def pixels(self) -> np.ndarray:
        """Return the (21, 2) landmark positions in pixels (x * width, y * height)."""
        xy = self.landmarks[:, :2].astype(float)
        return xy * np.array([self.width, self.height], dtype=float)
