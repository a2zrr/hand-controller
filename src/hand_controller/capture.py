"""Webcam capture with a clear error when the camera cannot be opened."""

from __future__ import annotations

import sys

import cv2
import numpy as np

from hand_controller.config import CameraConfig


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened."""


class Camera:
    """Context manager around cv2.VideoCapture."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> Camera:
        self._cap = cv2.VideoCapture(self.cfg.index)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(_camera_help(self.cfg.index))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if no frame was delivered."""
        if self._cap is None:
            raise RuntimeError("Camera is not open")
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def size(self) -> tuple[int, int]:
        """Actual (width, height) of the frames the camera delivers."""
        if self._cap is None:
            return self.cfg.width, self.cfg.height
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def _camera_help(index: int) -> str:
    """Build the error message shown when the camera cannot be opened."""
    hints = [
        f"Could not open camera index {index}.",
        "  - Another app may be using the webcam (close Zoom, browsers, OBS, ...).",
        "  - Try another index: `uv run hand-controller --camera 1` (or 2, 3...).",
    ]
    if sys.platform.startswith("win"):
        hints.append("  - Windows: Settings > Privacy & security > Camera > allow desktop apps to use the camera.")
    else:
        hints.append("  - Linux: check the device exists (`ls /dev/video*`) and that you are in the `video` group.")
    return "\n".join(hints)
