"""Feature extraction. The ONE feature function used by recorder, trainer and runtime.

Features are computed from the landmarks of the mirrored camera image:
    1. convert normalized x/y to pixels (x * width, y * height), because the two axes
       have different sizes and distances would be distorted otherwise;
    2. subtract the wrist position (landmark 0);
    3. divide by the palm size (distance from wrist to middle MCP, landmark 9).
This makes the features independent of hand position and distance to the camera.
"""

from __future__ import annotations

import numpy as np

from hand_controller.core import Hand

FEATURE_VERSION = 1
NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9


def feature_names(use_z: bool) -> list[str]:
    """Return the column names matching the output of `compute_features`."""
    names: list[str] = []
    for i in range(NUM_LANDMARKS):
        names += [f"x{i}", f"y{i}"]
        if use_z:
            names.append(f"z{i}")
    return names


def compute_features(landmarks: np.ndarray, width: int, height: int, use_z: bool = False) -> np.ndarray:
    """Turn (21, 3) normalized landmarks into a flat feature vector.

    Args:
        landmarks: Array of shape (21, 3) with normalized x, y, z.
        width: Frame width in pixels.
        height: Frame height in pixels.
        use_z: Whether to append the depth value of each landmark.

    Returns:
        1-D float array, 42 values (or 63 with use_z).
    """
    lm = np.asarray(landmarks, dtype=float)
    pts = lm[:, :2] * np.array([width, height], dtype=float)
    rel = pts - pts[WRIST]
    palm = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))
    if palm < 1e-6:
        palm = 1.0  # degenerate detection; features become large but finite
    columns = [rel / palm]
    if use_z:
        # z uses the same units as x (fraction of the width), so convert it to pixels first.
        z_rel = (lm[:, 2] - lm[WRIST, 2]) * width / palm
        columns.append(z_rel[:, None])
    return np.concatenate(columns, axis=1).reshape(-1)


def hand_features(hand: Hand, use_z: bool = False) -> np.ndarray:
    """Convenience wrapper: features for a detected `Hand`."""
    return compute_features(hand.landmarks, hand.width, hand.height, use_z)
