"""Build synthetic hand landmark sets for tests. No camera or MediaPipe model needed.

The hand is drawn in pixel space with the wrist at the bottom centre of a 640x480 frame.
Each finger is a short chain of points along a direction `u`: extended fingers point along
`u` and are long, curled fingers fold back towards the palm.
"""

from __future__ import annotations

import math

import numpy as np

from hand_controller.core import Hand

WIDTH = 640
HEIGHT = 480
WRIST_PX = np.array([320.0, 400.0])
# Lateral offsets of index, middle, ring, pinky MCP joints (pixels, perpendicular to the hand).
LATERAL = [-30.0, -10.0, 10.0, 30.0]
FINGER_CHAINS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]


def direction_vector(angle_deg: float) -> np.ndarray:
    """Unit vector in IMAGE pixel space for a math angle (0 = right, 90 = up)."""
    rad = math.radians(angle_deg)
    return np.array([math.cos(rad), -math.sin(rad)])


def make_hand(angle_deg: float, extended: list[bool], score: float = 0.95) -> Hand:
    """Build a Hand whose fingers are extended (True) or curled (False).

    Args:
        angle_deg: Direction the hand points (math convention, 0 = right, 90 = up).
        extended: Four flags for index, middle, ring, pinky.
        score: Handedness score to attach to the hand.
    """
    u = direction_vector(angle_deg)
    perp = np.array([-u[1], u[0]])
    pts = np.zeros((21, 2))
    pts[0] = WRIST_PX
    # Thumb: points off to one side, its value does not matter for the classifier.
    pts[1:5] = WRIST_PX - 40.0 * perp + np.outer([0.2, 0.5, 0.8, 1.0], u) * 30.0

    for finger, (mcp_i, pip_i, dip_i, tip_i) in enumerate(FINGER_CHAINS):
        mcp = WRIST_PX + 100.0 * u + LATERAL[finger] * perp
        pts[mcp_i] = mcp
        if extended[finger]:
            pts[pip_i] = mcp + 40.0 * u
            pts[dip_i] = mcp + 65.0 * u
            pts[tip_i] = mcp + 90.0 * u
        else:
            pts[pip_i] = mcp + 30.0 * u
            pts[dip_i] = mcp + 12.0 * u
            pts[tip_i] = mcp - 5.0 * u

    landmarks = np.zeros((21, 3))
    landmarks[:, 0] = pts[:, 0] / WIDTH
    landmarks[:, 1] = pts[:, 1] / HEIGHT
    return Hand(landmarks=landmarks, handedness="Right", score=score, width=WIDTH, height=HEIGHT)


def pointing(angle_deg: float) -> Hand:
    """Index finger extended, others curled."""
    return make_hand(angle_deg, [True, False, False, False])


def open_palm() -> Hand:
    """All four fingers extended."""
    return make_hand(90.0, [True, True, True, True])


def fist() -> Hand:
    """All four fingers curled."""
    return make_hand(90.0, [False, False, False, False])
