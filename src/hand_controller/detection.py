"""Wrapper around MediaPipe's HandLandmarker (Tasks API, VIDEO running mode)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import mediapipe as mp
import numpy as np

from hand_controller.config import DetectionConfig, resolve_project_path
from hand_controller.core import Hand

MODEL_HELP = (
    "Hand model not found: {path}\n"
    "Download it once with: uv run python tools/download_model.py"
)


@dataclass(frozen=True)
class FramePlan:
    """How a BGR camera frame is converted before MediaPipe reads it."""

    conversion: int  # cv2 colour conversion code
    image_format: mp.ImageFormat  # must have as many channels as the conversion output


def frame_plan_for(delegate: str) -> FramePlan:
    """Return the frame format for a concrete delegate ("gpu" or "cpu").

    This is the only place that decides it. The GPU delegate on macOS (Metal) aborts the
    process with "unsupported ImageFrame format: 1" when given 3-channel SRGB images, and an
    abort cannot be caught, so the GPU path always gets 4-channel SRGBA. The CPU path keeps SRGB.
    """
    if delegate == "gpu":
        return FramePlan(conversion=cv2.COLOR_BGR2RGBA, image_format=mp.ImageFormat.SRGBA)
    if delegate == "cpu":
        return FramePlan(conversion=cv2.COLOR_BGR2RGB, image_format=mp.ImageFormat.SRGB)
    raise ValueError(f"No frame format for delegate {delegate!r}; expected 'gpu' or 'cpu'")


def _create_landmarker(path: Path, cfg: DetectionConfig) -> tuple[mp.tasks.vision.HandLandmarker, str]:
    """Create the landmarker on the requested delegate.

    "auto" tries the GPU first, because it is about 3x faster on Linux, and falls back to CPU
    when the GPU delegate is not available (for example on some Windows setups).

    Returns the landmarker and the concrete delegate it was created on ("gpu" or "cpu").
    """
    delegates = {"cpu": ["cpu"], "gpu": ["gpu"], "auto": ["gpu", "cpu"]}
    if cfg.delegate not in delegates:
        raise ValueError("detection.delegate must be 'auto', 'gpu' or 'cpu'")
    last_error: Exception | None = None
    for name in delegates[cfg.delegate]:
        delegate = mp.tasks.BaseOptions.Delegate.GPU if name == "gpu" else mp.tasks.BaseOptions.Delegate.CPU
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path), delegate=delegate),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        try:
            landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # the delegate may be unsupported on this machine
            last_error = exc
            continue
        print(f"Hand detector using the {name.upper()} delegate")
        return landmarker, name
    raise RuntimeError(f"Could not create the hand detector on delegate '{cfg.delegate}': {last_error}")


class HandDetector:
    """Detects at most one hand per frame. Use as a context manager to release the model."""

    def __init__(self, cfg: DetectionConfig) -> None:
        path = resolve_project_path(cfg.model_path)
        if not path.exists():
            raise FileNotFoundError(MODEL_HELP.format(path=path))
        self._landmarker, delegate = _create_landmarker(path, cfg)
        # The frame format follows the delegate that was actually created, not the configured one.
        self._frame_plan = frame_plan_for(delegate)
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
        converted = cv2.cvtColor(frame_bgr, self._frame_plan.conversion)
        image = mp.Image(image_format=self._frame_plan.image_format, data=np.ascontiguousarray(converted))
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
