"""Wrapper around MediaPipe's HandLandmarker (Tasks API, VIDEO running mode)."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2
import mediapipe as mp
import numpy as np

from hand_controller.config import DetectionConfig
from hand_controller.core import Hand

MODEL_HELP = (
    "Hand model not found: {path}\n"
    "Download it once with: uv run python tools/download_model.py"
)


class HandDetector:
    """Detects at most one hand per frame. Use as a context manager to release the model."""

    def __init__(self, cfg: DetectionConfig) -> None:
        path = Path(cfg.model_path)
        if not path.exists():
            raise FileNotFoundError(MODEL_HELP.format(path=path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        self._swap_handedness = cfg.swap_handedness

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the MediaPipe graph."""
        self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Hand | None:
        """Run detection on one BGR frame.

        The frame must already be mirrored if you want mirrored landmarks: this class
        does not flip the image itself. VIDEO mode needs strictly increasing timestamps,
        so a non-increasing timestamp is bumped by 1 ms.

        Returns:
            The detected hand, or None when no hand is visible.
        """
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=float
        )
        category = result.handedness[0][0]
        label = category.category_name
        if self._swap_handedness:
            label = "Left" if label == "Right" else "Right"
        return Hand(
            landmarks=landmarks,
            handedness=label,
            score=float(category.score),
            width=width,
            height=height,
        )
