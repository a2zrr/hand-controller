"""Recorded samples: CSV reading/writing and the session-based train/test split.

File format (one CSV per recording session in data/):
    label,session,width,height,lm0_x,lm0_y,lm0_z,...,lm20_x,lm20_y,lm20_z
Landmarks are the normalized MediaPipe values from the mirrored image. Features are NOT
stored; they are computed from the landmarks by `features.compute_features`, the same
function the runtime uses, so the training data can never drift from the runtime.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hand_controller.core import Hand, State
from hand_controller.features import NUM_LANDMARKS, compute_features

LANDMARK_COLUMNS = [f"lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
HEADER = ["label", "session", "width", "height", *LANDMARK_COLUMNS]
LABELS = [s.value for s in State]


@dataclass(frozen=True)
class Sample:
    """One recorded hand pose with its label and the session it came from."""

    label: str
    session: str
    width: int
    height: int
    landmarks: np.ndarray  # (21, 3)

    def to_hand(self) -> Hand:
        """Convert back to a Hand so the rule classifier can be evaluated on recorded data."""
        return Hand(self.landmarks, "Right", 1.0, self.width, self.height)


class SampleWriter:
    """Appends samples to one session file. Each row is flushed immediately, so a crash loses nothing."""

    def __init__(self, path: Path, session: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.session = session
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(HEADER)
        self._fh.flush()

    def write(self, label: str, hand: Hand) -> None:
        """Append one labelled sample."""
        row = [label, self.session, hand.width, hand.height, *hand.landmarks.reshape(-1).round(6).tolist()]
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        """Close the file."""
        self._fh.close()


def load_samples(paths: Iterable[Path]) -> list[Sample]:
    """Read every sample from the given CSV files."""
    samples: list[Sample] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                values = np.array([float(row[c]) for c in LANDMARK_COLUMNS]).reshape(NUM_LANDMARKS, 3)
                samples.append(
                    Sample(row["label"], row["session"], int(row["width"]), int(row["height"]), values)
                )
    return samples


def load_data_dir(data_dir: Path) -> list[Sample]:
    """Load all session CSV files in a folder. Raises if there are none."""
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No recorded CSV files in {data_dir}. Record some with `uv run python tools/record.py`.")
    return load_samples(paths)


def to_features(samples: list[Sample], use_z: bool) -> np.ndarray:
    """Feature matrix, one row per sample, built with the shared feature function."""
    return np.array(
        [compute_features(s.landmarks, s.width, s.height, use_z) for s in samples],
        dtype=float,
    )


def split_by_session(samples: list[Sample], test_fraction: float = 0.25) -> tuple[list[Sample], list[Sample]]:
    """Split whole sessions into train and test, so frames of one session never appear in both.

    The last sessions (in sorted name order) become the test set, at least one of them.
    Requires at least two sessions.
    """
    sessions = sorted({s.session for s in samples})
    if len(sessions) < 2:
        raise ValueError(
            f"Need at least 2 recording sessions to split train/test by session, found {len(sessions)}. "
            "Record another session (run tools/record.py again)."
        )
    n_test = max(1, round(len(sessions) * test_fraction))
    test_sessions = set(sessions[-n_test:])
    train = [s for s in samples if s.session not in test_sessions]
    test = [s for s in samples if s.session in test_sessions]
    return train, test
